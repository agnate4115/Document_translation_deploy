"""
Simplified translator module - Azure OpenAI only.
"""

import logging
import os
import re
import unicodedata
from copy import copy
from string import Template
from typing import cast

import openai

from pdf2zh.cache import TranslationCache
from pdf2zh.config import ConfigManager

from tenacity import retry, retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential


logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    """Base class for translators."""
    name = "base"
    envs = {}
    lang_map: dict[str, str] = {}
    CustomPrompt = False

    def __init__(self, lang_in: str, lang_out: str, model: str, ignore_cache: bool):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model
        self.ignore_cache = ignore_cache

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        self.envs = copy(self.envs)
        if ConfigManager.get_translator_by_name(self.name):
            self.envs = ConfigManager.get_translator_by_name(self.name)
        needUpdate = False
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
                needUpdate = True
        if needUpdate:
            ConfigManager.set_translator_by_name(self.name, self.envs)
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]
            ConfigManager.set_translator_by_name(self.name, self.envs)

    def add_cache_impact_parameters(self, k: str, v):
        self.cache.add_params(k, v)

    def translate(self, text: str, ignore_cache: bool = False) -> str:
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = self.do_translate(text)
        translation = self._postprocess(text, translation)

        self.cache.set(text, translation)
        return translation

    def _postprocess(self, source: str, translation: str) -> str:
        """Validate and fix translation output."""
        if not translation or not source.strip():
            return translation

        src_markers = re.findall(r"\{v\d+\}", source)
        if not src_markers:
            return translation

        src_marker_set = set(src_markers)
        tgt_marker_set = set(re.findall(r"\{v\d+\}", translation))

        # Retry up to 2 times if markers are missing
        missing = src_marker_set - tgt_marker_set
        retries = 0
        while missing and retries < 2:
            logger.warning(f"Missing formula markers {missing}, retry {retries + 1}...")
            translation = self.do_translate(source)
            tgt_marker_set = set(re.findall(r"\{v\d+\}", translation))
            missing = src_marker_set - tgt_marker_set
            retries += 1

        # If markers are STILL missing after retries, append them to preserve layout
        if missing:
            logger.warning(f"Could not recover markers {missing} after retries, appending them")
            for m in sorted(missing):
                translation += f" {m}"

        # Remove any extra markers that the LLM hallucinated
        extra = tgt_marker_set - src_marker_set
        if extra:
            for m in extra:
                translation = translation.replace(m, "", 1)
            translation = re.sub(r"  +", " ", translation).strip()

        return translation

    def do_translate(self, text: str) -> str:
        raise NotImplementedError

    def prompt(
        self, text: str, prompt_template: Template | None = None
    ) -> list[dict[str, str]]:
        try:
            return [
                {
                    "role": "user",
                    "content": cast(Template, prompt_template).safe_substitute(
                        {
                            "lang_in": self.lang_in,
                            "lang_out": self.lang_out,
                            "text": text,
                        }
                    ),
                }
            ]
        except AttributeError:
            pass
        except Exception:
            logging.exception("Error parsing prompt, use the default prompt.")

        # Count formula markers in the source text for explicit instruction
        markers = re.findall(r"\{v\d+\}", text)
        marker_instruction = ""
        if markers:
            marker_instruction = (
                f"\n\nIMPORTANT: The source contains {len(markers)} formula placeholder(s): {', '.join(markers)}. "
                "You MUST include every single one in your translation, exactly as written. "
                "Do NOT translate, modify, reorder, or omit any of them."
            )

        return [
            {
                "role": "system",
                "content": (
                    "You are a professional document translation engine. Follow these rules strictly:\n"
                    "1. Translate EVERY word completely — never skip, summarize, abbreviate, or omit any content.\n"
                    "2. Preserve ALL formula/variable placeholders exactly: {v0}, {v1}, {v2}, etc. Copy them character-for-character.\n"
                    "3. Translate ALL table cell content — every cell, header, and label must be translated.\n"
                    "4. Keep numbers, units, dates, proper nouns, and technical terms accurate.\n"
                    "5. Maintain the exact same paragraph structure.\n"
                    "6. Output ONLY the translated text — no explanations, notes, labels, or extra text.\n"
                    "7. Do NOT add or remove any {v*} placeholders. The count must match exactly.\n"
                    "8. If a word has no direct translation, transliterate it phonetically.\n"
                    "9. Do NOT prefix your output with 'Translation:' or 'Translated Text:' — output the translation directly.\n"
                    "10. Short text (1-3 words) must still be fully translated, not left unchanged.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Translate from {self.lang_in} to {self.lang_out}. "
                    "Translate every word. Output only the translation."
                    f"{marker_instruction}\n\n"
                    f"{text}"
                ),
            },
        ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, id: int):
        return f"<b{id}>"

    def get_rich_text_right_placeholder(self, id: int):
        return f"</b{id}>"

    def get_formular_placeholder(self, id: int):
        return self.get_rich_text_left_placeholder(
            id
        ) + self.get_rich_text_right_placeholder(id)


class AzureOpenAITranslator(BaseTranslator):
    """Azure OpenAI translator implementation."""
    name = "azure-openai"
    envs = {
        "AZURE_OPENAI_BASE_URL": None,  # e.g. "https://xxx.openai.azure.com"
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_MODEL": "gpt-4o-mini",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",  # default api version
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = self.envs["AZURE_OPENAI_BASE_URL"]
        if not model:
            model = self.envs["AZURE_OPENAI_MODEL"]
        api_version = self.envs.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if api_key is None:
            api_key = self.envs["AZURE_OPENAI_API_KEY"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {
            "temperature": 0,       # Deterministic output to preserve formula markers
            "max_tokens": 4096,     # Prevent truncation of longer paragraphs
        }
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key,
        )
        self.prompttext = prompt
        self._last_source = ""  # Track source text for postprocessing
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        
        # Filter for think tags if present
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"API error, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text) -> str:
        self._last_source = text
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        if not response.choices:
            if hasattr(response, "error"):
                raise ValueError("Error response from Service", response.error)
            raise ValueError("Empty response from translation service")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty content in translation response")
        content = content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        # Remove common LLM artifacts - prefixes the model sometimes adds
        for prefix in [
            "Translated Text:", "Translation:", "Translated text:",
            "Here is the translation:", "Here's the translation:",
            "Output:", "Result:",
        ]:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        # Remove wrapping quotes the model sometimes adds
        if len(content) > 2 and content[0] == '"' and content[-1] == '"':
            inner = content[1:-1]
            # Only strip if the source didn't have quotes
            if not (self._last_source and self._last_source.startswith('"')):
                content = inner
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)
