"""
Streamlit application for PDF translation using Azure OpenAI.
Preserves all PDF formatting including fonts, sizes, colors, styles, images, tables, layouts, headers, footers, spacing, alignment, and positioning.
"""

import os
import base64
from pathlib import Path
import uuid
import streamlit as st
from streamlit import components
from dotenv import load_dotenv
import logging

from pdf2zh.high_level import translate_stream, download_remote_fonts
from pdf2zh.doclayout import OnnxModel, ModelInstance
from pdf2zh.translator import AzureOpenAITranslator

# Load environment variables
load_dotenv()

# Helper function to get env var and strip quotes if present
def get_env_var(key, default=""):
    """Get environment variable and strip quotes if present."""
    value = os.getenv(key, default)
    if isinstance(value, str) and len(value) >= 2:
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
    # Remove trailing slash from endpoint URLs
    if key in ["AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT"] and value.endswith("/"):
        value = value.rstrip("/")
    return value

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel("CRITICAL")
logging.getLogger("openai").setLevel("CRITICAL")
logging.getLogger("httpcore").setLevel("CRITICAL")

# Initialize session state
if "translated_pdf" not in st.session_state:
    st.session_state.translated_pdf = None
if "original_pdf" not in st.session_state:
    st.session_state.original_pdf = None
if "original_filename" not in st.session_state:
    st.session_state.original_filename = ""
if "translation_progress" not in st.session_state:
    st.session_state.translation_progress = 0
if "translation_status" not in st.session_state:
    st.session_state.translation_status = ""
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

# Page configuration
st.set_page_config(
    page_title="PDF Translator ",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ONNX model (cache it in session state)
@st.cache_resource
def load_model():
    """Load and cache the ONNX model for document layout detection."""
    try:
        return OnnxModel.load_available()
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        st.info("Please ensure all dependencies are installed correctly.")
        return None

# Load model
ModelInstance.value = load_model()

if ModelInstance.value is None:
    st.error("Failed to load document layout model. Please check your installation.")
    st.stop()

# Load Azure OpenAI credentials from environment (not editable in UI)
azure_endpoint = get_env_var("AZURE_OPENAI_BASE_URL", get_env_var("AZURE_OPENAI_ENDPOINT", ""))
azure_api_key = get_env_var("AZURE_OPENAI_API_KEY", "")
deployment_name = get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
api_version = get_env_var("AZURE_OPENAI_API_VERSION", "2024-06-01")

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# Simple password gate for translation actions
st.sidebar.subheader("Access")
pw = st.sidebar.text_input("Password", type="password", placeholder="Enter password to unlock translation")
if pw:
    st.session_state.auth_ok = pw == "Epassword@_4"
if not st.session_state.auth_ok:
    st.sidebar.warning("Translation is locked. Enter the password to unlock.")

# Azure OpenAI info (no key/URI values shown)
st.sidebar.subheader("Azure ")
st.sidebar.markdown(
    "- **Credentials source**: `.env`\n"
    "- **UI note**: Secrets/URLs are never shown here",
)

# Language Selection (only when unlocked)
LANG_OPTIONS = [
    ("English", "en"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Chinese (Traditional)", "zh-TW"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Russian", "ru"),
    ("Arabic", "ar"),
]
lang_labels = [name for name, _ in LANG_OPTIONS]
label_to_code = {name: code for name, code in LANG_OPTIONS}

if st.session_state.auth_ok:
    st.sidebar.subheader("Language Settings")
    lang_in_label = st.sidebar.selectbox(
        "Source Language",
        options=lang_labels,
        index=0,
        help="Language of the source PDF",
    )
    lang_out_label = st.sidebar.selectbox(
        "Target Language",
        options=lang_labels,
        index=1,  # default target different from source
        help="Language to translate to",
    )
    lang_in = label_to_code[lang_in_label]
    lang_out = label_to_code[lang_out_label]

    # Translation Options
    st.sidebar.subheader("Translation Options")
    threads = st.sidebar.slider(
        "Translation Threads",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of parallel translation threads",
    )
    skip_subset_fonts = st.sidebar.checkbox(
        "Skip Font Subsetting",
        value=False,
        help="Skip font subsetting (may increase file size but improve compatibility)",
    )
else:
    # Safe defaults (unused until unlocked)
    lang_in = "en"
    lang_out = "zh-CN"
    threads = 4
    skip_subset_fonts = False

# Validate Azure credentials
def validate_azure_credentials():
    """Validate that Azure OpenAI credentials are provided."""
    if not azure_endpoint:
        return False, "Azure Endpoint is required"
    if not azure_api_key:
        return False, "API Key is required"
    if not deployment_name:
        return False, "Deployment Name is required"
    if not api_version:
        return False, "API Version is required"
    return True, ""

# Helper to render a PDF inline using Syncfusion PDF Viewer
def render_pdf(title: str, pdf_bytes: bytes):
    if not pdf_bytes:
        return

    # Encode PDF as base64 for Syncfusion viewer
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    container_id = f"pdf-container-{uuid.uuid4().hex}"

    # Read Syncfusion license key from environment (required for licensed usage)
    # NOTE: Set SYNCFUSION_LICENSE_KEY in your Streamlit deployment environment.
    syncfusion_key = os.getenv("SYNCFUSION_LICENSE_KEY", "")

    pdf_display = f"""
    <div style="border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 10px; overflow: hidden;">
      <div style="padding: 10px 12px; background: rgba(49, 51, 63, 0.06); font-weight: 600;">{title}</div>
      <div id="{container_id}" style="width: 100%; height: 820px;"></div>
    </div>
    <script type="text/javascript">
      (function() {{
        var b64Data = "{b64}";
        var licenseKey = "{syncfusion_key}";
        var containerId = "{container_id}";

        function initPdfViewer() {{
          if (!window.ej || !ej.pdfviewer || !ej.pdfviewer.PdfViewer) {{
            console.error("Syncfusion EJ2 PDF Viewer scripts not loaded.");
            var c = document.getElementById(containerId);
            if (c) {{
              c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Failed to load PDF viewer. Please check network access to cdn.syncfusion.com.</div>";
            }}
            return;
          }}

          try {{
            if (licenseKey) {{
              ej.base.registerLicense(licenseKey);
            }}

            ej.pdfviewer.PdfViewer.Inject(
              ej.pdfviewer.Toolbar,
              ej.pdfviewer.Magnification,
              ej.pdfviewer.Navigation,
              ej.pdfviewer.Print,
              ej.pdfviewer.TextSelection,
              ej.pdfviewer.TextSearch,
              ej.pdfviewer.Annotation,
              ej.pdfviewer.FormFields,
              ej.pdfviewer.FormDesigner
            );

            var viewer = new ej.pdfviewer.PdfViewer({{
              enableToolbar: true,
              enableNavigation: true,
              enableTextSelection: true,
              enableAnnotation: true,
              width: "100%",
              height: "100%",
              serviceUrl: "https://ej2services.syncfusion.com/production/web-services/api/pdfviewer"
            }});

            viewer.appendTo("#" + containerId);
            viewer.load("data:application/pdf;base64," + b64Data, null);
          }} catch (e) {{
            console.error("Error initializing Syncfusion PDF Viewer", e);
            var c = document.getElementById(containerId);
            if (c) {{
              c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Unable to preview PDF inline. Please use the download button below to open it.</div>";
            }}
          }}
        }}

        function loadSyncfusionResources(callback) {{
          // Load CSS
          var link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = "https://cdn.syncfusion.com/ej2/26.1.35/material.css";
          document.head.appendChild(link);

          // Load JS bundle
          var script = document.createElement("script");
          script.src = "https://cdn.syncfusion.com/ej2/26.1.35/dist/ej2.min.js";
          script.onload = callback;
          script.onerror = function() {{
            console.error("Failed to load Syncfusion EJ2 script from CDN.");
            var c = document.getElementById(containerId);
            if (c) {{
              c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Failed to load Syncfusion viewer scripts from CDN. Please check your network.</div>";
            }}
          }};
          document.head.appendChild(script);
        }}

        if (!window.ej || !ej.pdfviewer || !ej.pdfviewer.PdfViewer) {{
          loadSyncfusionResources(initPdfViewer);
        }} else {{
          initPdfViewer();
        }}
      }})();
    </script>
    """

    components.v1.html(pdf_display, height=860, scrolling=False)

# Main content area
st.title("PDF Translator")
st.caption(
    "Upload a PDF, translate it, then preview **original vs translated** side-by-side right here."
)

st.markdown("### Upload")
uploaded_file = st.file_uploader("PDF file", type=["pdf"])

# Display uploaded file info
if uploaded_file is not None:
    st.info(f"📎 Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")

st.markdown("### Translate")
action_col, status_col = st.columns([1, 3], vertical_alignment="center")
with action_col:
    translate_button = st.button(
        "Translate",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.auth_ok,
    )
with status_col:
    if st.session_state.translation_status:
        if "Error" in st.session_state.translation_status:
            st.error(st.session_state.translation_status)
        else:
            st.info(st.session_state.translation_status)
    if not st.session_state.auth_ok:
        st.warning("Enter the password in the sidebar to enable translation.")

# Progress bar
if st.session_state.translation_progress > 0:
    progress_bar = st.progress(st.session_state.translation_progress)
    st.caption(f"Progress: {st.session_state.translation_progress * 100:.1f}%")

# Translation function
def translate_pdf(pdf_bytes, azure_endpoint, azure_api_key, deployment_name, api_version, lang_in, lang_out, threads, skip_subset_fonts):
    """Translate PDF using Azure OpenAI."""
    try:
        # Validate credentials
        is_valid, error_msg = validate_azure_credentials()
        if not is_valid:
            st.error(f"Configuration Error: {error_msg}")
            return None

        # Update progress
        st.session_state.translation_progress = 0.1
        st.session_state.translation_status = "Reading PDF file..."

        st.session_state.translation_progress = 0.2
        st.session_state.translation_status = "Initializing translator..."

        # Create Azure OpenAI translator environment variables
        envs = {
            "AZURE_OPENAI_BASE_URL": azure_endpoint,
            "AZURE_OPENAI_API_KEY": azure_api_key,
            "AZURE_OPENAI_MODEL": deployment_name,
            "AZURE_OPENAI_API_VERSION": api_version,
        }

        st.session_state.translation_progress = 0.3
        st.session_state.translation_status = "Downloading fonts..."

        # Download fonts for target language
        try:
            font_path = download_remote_fonts(lang_out.lower())
            if font_path is None:
                st.info("Using default fonts for translation.")
        except Exception as e:
            st.info(f"Font download note: {e}. Using default fonts - translation will continue.")

        st.session_state.translation_progress = 0.4
        st.session_state.translation_status = "Translating PDF (this may take a while)..."

        # Progress callback
        def progress_callback(tqdm_obj):
            """Update progress from tqdm."""
            if hasattr(tqdm_obj, 'n') and hasattr(tqdm_obj, 'total') and tqdm_obj.total > 0:
                progress = 0.4 + (tqdm_obj.n / tqdm_obj.total) * 0.5
                st.session_state.translation_progress = min(progress, 0.9)
                desc = getattr(tqdm_obj, 'desc', 'Translating...')
                st.session_state.translation_status = desc if desc else "Translating pages..."

        # Translate PDF
        doc_mono_bytes, doc_dual_bytes = translate_stream(
            pdf_bytes,
            lang_in=lang_in,
            lang_out=lang_out,
            service="azure-openai",
            thread=threads,
            callback=progress_callback,
            model=ModelInstance.value,
            envs=envs,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=False,
        )

        st.session_state.translation_progress = 1.0
        st.session_state.translation_status = "Translation complete!"

        return doc_mono_bytes, doc_dual_bytes

    except Exception as e:
        st.session_state.translation_status = f"Error: {str(e)}"
        st.error(f"Translation failed: {str(e)}")
        logging.exception("Translation error")
        return None

# Handle translation
if translate_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a PDF file first.")
    else:
        # Reset progress
        st.session_state.translation_progress = 0
        st.session_state.translation_status = "Starting translation..."
        # Cache original PDF bytes in session for preview
        if (
            st.session_state.original_pdf is None
            or st.session_state.original_filename != uploaded_file.name
        ):
            pdf_bytes = uploaded_file.read()
            st.session_state.original_pdf = pdf_bytes
            st.session_state.original_filename = uploaded_file.name
        else:
            pdf_bytes = st.session_state.original_pdf

        # Translate
        result = translate_pdf(
            pdf_bytes,
            azure_endpoint,
            azure_api_key,
            deployment_name,
            api_version,
            lang_in,
            lang_out,
            threads,
            skip_subset_fonts
        )

        if result:
            doc_mono_bytes, doc_dual_bytes = result
            st.session_state.translated_pdf = {
                "mono": doc_mono_bytes,
                "dual": doc_dual_bytes,
                "filename": Path(uploaded_file.name).stem
            }
            st.success("✅ Translation completed successfully!")

# Preview (always inline) + download translated PDFs
st.markdown("### Preview")
preview_mode = "Translated (monolingual)"
if st.session_state.translated_pdf:
    preview_mode = st.radio(
        "Translated preview",
        options=["Translated (monolingual)", "Translated (bilingual)"],
        horizontal=True,
        label_visibility="visible",
    )

left, right = st.columns(2, gap="large")
with left:
    if uploaded_file is not None and st.session_state.original_pdf:
        render_pdf("Original PDF", st.session_state.original_pdf)
    else:
        st.info("Upload a PDF to preview it here.")

with right:
    if st.session_state.translated_pdf:
        if preview_mode == "Translated (bilingual)":
            render_pdf("Translated PDF (bilingual)", st.session_state.translated_pdf["dual"])
        else:
            render_pdf("Translated PDF (monolingual)", st.session_state.translated_pdf["mono"])
    else:
        st.info("Translate the PDF to preview the result here.")

if st.session_state.translated_pdf:
    st.markdown("### Download")
    col_dl_mono, col_dl_dual = st.columns(2)
    with col_dl_mono:
        st.download_button(
            label="Download translated PDF (monolingual)",
            data=st.session_state.translated_pdf["mono"],
            file_name=f"{st.session_state.translated_pdf['filename']}-mono.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_dl_dual:
        st.download_button(
            label="Download translated PDF (bilingual)",
            data=st.session_state.translated_pdf["dual"],
            file_name=f"{st.session_state.translated_pdf['filename']}-dual.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>PDF Translator </p>
        <p>Preserves fonts, sizes, colors, styles, images, tables, layouts, headers, footers, spacing, alignment, and positioning</p>
    </div>
    """,
    unsafe_allow_html=True
)
