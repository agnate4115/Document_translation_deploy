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


def get_env_var(key, default=""):
    """Get environment variable and strip quotes if present."""
    value = os.getenv(key, default)
    if isinstance(value, str) and len(value) >= 2:
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
    if key in ["AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT"] and value.endswith("/"):
        value = value.rstrip("/")
    return value


# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel("CRITICAL")
logging.getLogger("openai").setLevel("CRITICAL")
logging.getLogger("httpcore").setLevel("CRITICAL")

# Initialize session state
for key, default in {
    "translated_pdf": None,
    "original_pdf": None,
    "original_filename": "",
    "translation_progress": 0,
    "translation_status": "",
    "auth_ok": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Page configuration
st.set_page_config(
    page_title="DocTranslate - Professional Document Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 50%, #1e3a5f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1rem;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #2d5f8a;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f0f6fc;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #1e3a5f;
        background: #e3eef8;
    }

    /* Stats card */
    .stat-card {
        background: white;
        border: 1px solid #e0e4e8;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* Language selector cards */
    .lang-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }

    /* Feature badges */
    .feature-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1e3a5f;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Download buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #15304f 0%, #245178 100%) !important;
    }

    /* Progress section */
    .progress-section {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e0e4e8;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ONNX model
@st.cache_resource
def load_model():
    """Load and cache the ONNX model for document layout detection."""
    try:
        return OnnxModel.load_available()
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

ModelInstance.value = load_model()
if ModelInstance.value is None:
    st.error("Failed to load document layout model. Please check your installation.")
    st.stop()

# Load Azure OpenAI credentials
azure_endpoint = get_env_var("AZURE_OPENAI_BASE_URL", get_env_var("AZURE_OPENAI_ENDPOINT", ""))
azure_api_key = get_env_var("AZURE_OPENAI_API_KEY", "")
deployment_name = get_env_var("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
api_version = get_env_var("AZURE_OPENAI_API_VERSION", "2024-06-01")

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Settings")

    # Password gate
    st.markdown("**Access**")
    pw = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")
    if pw:
        st.session_state.auth_ok = pw == "Epassword@_4"
    if not st.session_state.auth_ok:
        st.warning("Enter password to unlock translation.")
    else:
        st.success("Unlocked")

    st.divider()

    # Language Selection
    LANG_OPTIONS = [
        ("English", "en"),
        ("Hindi", "hi"),
        ("Chinese (Simplified)", "zh-CN"),
        ("Chinese (Traditional)", "zh-TW"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("French", "fr"),
        ("German", "de"),
        ("Spanish", "es"),
        ("Portuguese", "pt"),
        ("Italian", "it"),
        ("Russian", "ru"),
        ("Arabic", "ar"),
        ("Turkish", "tr"),
        ("Vietnamese", "vi"),
        ("Thai", "th"),
        ("Indonesian", "id"),
        ("Malay", "ms"),
        ("Bengali", "bn"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Marathi", "mr"),
        ("Urdu", "ur"),
        ("Gujarati", "gu"),
        ("Kannada", "kn"),
        ("Malayalam", "ml"),
        ("Ukrainian", "uk"),
        ("Polish", "pl"),
        ("Dutch", "nl"),
        ("Swedish", "sv"),
        ("Czech", "cs"),
        ("Romanian", "ro"),
        ("Greek", "el"),
        ("Hebrew", "iw"),
        ("Hungarian", "hu"),
        ("Danish", "da"),
        ("Finnish", "fi"),
        ("Norwegian", "no"),
    ]
    lang_labels = [name for name, _ in LANG_OPTIONS]
    label_to_code = {name: code for name, code in LANG_OPTIONS}

    if st.session_state.auth_ok:
        st.markdown("**Languages**")
        lang_in_label = st.selectbox(
            "Source Language",
            options=lang_labels,
            index=0,
        )
        lang_out_label = st.selectbox(
            "Target Language",
            options=lang_labels,
            index=1,  # Hindi by default
        )
        lang_in = label_to_code[lang_in_label]
        lang_out = label_to_code[lang_out_label]

        if lang_in == lang_out:
            st.error("Source and target languages must be different.")

        st.divider()

        st.markdown("**Advanced**")
        threads = st.slider("Translation Threads", 1, 8, 4, help="Parallel threads for faster translation")
        skip_subset_fonts = st.checkbox("Skip Font Subsetting", value=False, help="May increase file size but improve compatibility")
    else:
        lang_in, lang_out, threads, skip_subset_fonts = "en", "hi", 4, False

    st.divider()
    st.markdown(
        '<div style="text-align:center; color:#9ca3af; font-size:0.75rem;">'
        'Powered by Azure OpenAI<br>Credentials loaded from .env'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>DocTranslate</h1>
    <p>Professional document translation with 100% format preservation</p>
    <div style="margin-top: 1rem;">
        <span class="feature-badge">Tables</span>
        <span class="feature-badge">Images</span>
        <span class="feature-badge">Fonts</span>
        <span class="feature-badge">Layout</span>
        <span class="feature-badge">Headers & Footers</span>
        <span class="feature-badge">37+ Languages</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Upload Section
st.markdown("#### Upload Document")
uploaded_file = st.file_uploader(
    "Drop your PDF here or click to browse",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_size_kb = uploaded_file.size / 1024
    file_size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb / 1024:.1f} MB"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{uploaded_file.name[:20]}</div><div class="stat-label">File Name</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{file_size_str}</div><div class="stat-label">File Size</div></div>', unsafe_allow_html=True)
    with col3:
        if st.session_state.auth_ok:
            src_name = lang_in_label
            tgt_name = lang_out_label
        else:
            src_name = "English"
            tgt_name = "Hindi"
        st.markdown(f'<div class="stat-card"><div class="stat-value">{src_name} → {tgt_name}</div><div class="stat-label">Translation Direction</div></div>', unsafe_allow_html=True)

st.markdown("")

# Translate Button
col_btn, col_status = st.columns([1, 3], vertical_alignment="center")
with col_btn:
    translate_button = st.button(
        "Translate Document",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.auth_ok or (st.session_state.auth_ok and lang_in == lang_out),
    )
with col_status:
    if st.session_state.translation_status:
        if "Error" in st.session_state.translation_status:
            st.error(st.session_state.translation_status)
        elif "complete" in st.session_state.translation_status.lower():
            st.success(st.session_state.translation_status)
        else:
            st.info(st.session_state.translation_status)
    elif not st.session_state.auth_ok:
        st.info("Enter password in the sidebar to enable translation.")

# Progress bar
if 0 < st.session_state.translation_progress < 1:
    st.progress(st.session_state.translation_progress)
    st.caption(f"Progress: {st.session_state.translation_progress * 100:.0f}%")


# ─── PDF Viewer ────────────────────────────────────────────────────────────────

def render_pdf(title: str, pdf_bytes: bytes):
    if not pdf_bytes:
        return

    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    container_id = f"pdf-container-{uuid.uuid4().hex}"
    syncfusion_key = os.getenv("SYNCFUSION_LICENSE_KEY", "")

    pdf_display = f"""
    <div style="border: 1px solid rgba(49, 51, 63, 0.15); border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
      <div style="padding: 10px 14px; background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 100%); font-weight: 600; color: white; font-size: 0.9rem;">{title}</div>
      <div id="{container_id}" style="width: 100%; height: 820px;"></div>
    </div>
    <script type="text/javascript">
      (function() {{
        var b64Data = "{b64}";
        var licenseKey = "{syncfusion_key}";
        var containerId = "{container_id}";

        function initPdfViewer() {{
          if (!window.ej || !ej.pdfviewer || !ej.pdfviewer.PdfViewer) {{
            var c = document.getElementById(containerId);
            if (c) {{
              c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Failed to load PDF viewer.</div>";
            }}
            return;
          }}
          try {{
            if (licenseKey) {{ ej.base.registerLicense(licenseKey); }}
            ej.pdfviewer.PdfViewer.Inject(
              ej.pdfviewer.Toolbar, ej.pdfviewer.Magnification, ej.pdfviewer.Navigation,
              ej.pdfviewer.Print, ej.pdfviewer.TextSelection, ej.pdfviewer.TextSearch,
              ej.pdfviewer.Annotation, ej.pdfviewer.FormFields, ej.pdfviewer.FormDesigner
            );
            var viewer = new ej.pdfviewer.PdfViewer({{
              enableToolbar: true, enableNavigation: true, enableTextSelection: true,
              enableAnnotation: true, width: "100%", height: "100%",
              serviceUrl: "https://ej2services.syncfusion.com/production/web-services/api/pdfviewer"
            }});
            viewer.appendTo("#" + containerId);
            viewer.load("data:application/pdf;base64," + b64Data, null);
          }} catch (e) {{
            var c = document.getElementById(containerId);
            if (c) {{ c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Unable to preview PDF. Use the download button.</div>"; }}
          }}
        }}

        function loadSyncfusionResources(callback) {{
          var link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = "https://cdn.syncfusion.com/ej2/26.1.35/material.css";
          document.head.appendChild(link);
          var script = document.createElement("script");
          script.src = "https://cdn.syncfusion.com/ej2/26.1.35/dist/ej2.min.js";
          script.onload = callback;
          script.onerror = function() {{
            var c = document.getElementById(containerId);
            if (c) {{ c.innerHTML = "<div style='padding: 12px; color: #e63946;'>Failed to load viewer scripts from CDN.</div>"; }}
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


# ─── Credentials Validation ───────────────────────────────────────────────────

def validate_azure_credentials():
    if not azure_endpoint:
        return False, "Azure Endpoint is required"
    if not azure_api_key:
        return False, "API Key is required"
    if not deployment_name:
        return False, "Deployment Name is required"
    return True, ""


# ─── Translation ───────────────────────────────────────────────────────────────

def translate_pdf(pdf_bytes, azure_endpoint, azure_api_key, deployment_name, api_version, lang_in, lang_out, threads, skip_subset_fonts):
    """Translate PDF using Azure OpenAI."""
    try:
        is_valid, error_msg = validate_azure_credentials()
        if not is_valid:
            st.error(f"Configuration Error: {error_msg}")
            return None

        st.session_state.translation_progress = 0.1
        st.session_state.translation_status = "Reading PDF file..."

        st.session_state.translation_progress = 0.15
        st.session_state.translation_status = "Analyzing document layout..."

        envs = {
            "AZURE_OPENAI_BASE_URL": azure_endpoint,
            "AZURE_OPENAI_API_KEY": azure_api_key,
            "AZURE_OPENAI_MODEL": deployment_name,
            "AZURE_OPENAI_API_VERSION": api_version,
        }

        st.session_state.translation_progress = 0.25
        st.session_state.translation_status = "Downloading language fonts..."

        try:
            font_path = download_remote_fonts(lang_out.lower())
            if font_path is None:
                st.info("Using default fonts for translation.")
        except Exception as e:
            st.info(f"Font download note: {e}. Using default fonts.")

        st.session_state.translation_progress = 0.35
        st.session_state.translation_status = "Translating document (this may take a while)..."

        def progress_callback(tqdm_obj):
            if hasattr(tqdm_obj, 'n') and hasattr(tqdm_obj, 'total') and tqdm_obj.total > 0:
                progress = 0.35 + (tqdm_obj.n / tqdm_obj.total) * 0.55
                st.session_state.translation_progress = min(progress, 0.9)
                page_info = f"Page {tqdm_obj.n}/{tqdm_obj.total}"
                st.session_state.translation_status = f"Translating... {page_info}"

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
        st.warning("Please upload a PDF file first.")
    else:
        st.session_state.translation_progress = 0
        st.session_state.translation_status = "Starting translation..."
        if (
            st.session_state.original_pdf is None
            or st.session_state.original_filename != uploaded_file.name
        ):
            pdf_bytes = uploaded_file.read()
            st.session_state.original_pdf = pdf_bytes
            st.session_state.original_filename = uploaded_file.name
        else:
            pdf_bytes = st.session_state.original_pdf

        result = translate_pdf(
            pdf_bytes, azure_endpoint, azure_api_key, deployment_name,
            api_version, lang_in, lang_out, threads, skip_subset_fonts
        )

        if result:
            doc_mono_bytes, doc_dual_bytes = result
            st.session_state.translated_pdf = {
                "mono": doc_mono_bytes,
                "dual": doc_dual_bytes,
                "filename": Path(uploaded_file.name).stem
            }
            st.balloons()


# ─── Preview Section ───────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("#### Document Preview")

preview_mode = "Translated (monolingual)"
if st.session_state.translated_pdf:
    preview_mode = st.radio(
        "Preview mode",
        options=["Translated (monolingual)", "Translated (bilingual)"],
        horizontal=True,
    )

left, right = st.columns(2, gap="large")
with left:
    if uploaded_file is not None and st.session_state.original_pdf:
        render_pdf("Original Document", st.session_state.original_pdf)
    else:
        st.markdown(
            '<div style="border: 2px dashed #d1d5db; border-radius: 12px; padding: 4rem 2rem; text-align: center; color: #9ca3af;">'
            '<p style="font-size: 2rem; margin-bottom: 0.5rem;">📄</p>'
            '<p>Upload a PDF to preview it here</p>'
            '</div>',
            unsafe_allow_html=True,
        )

with right:
    if st.session_state.translated_pdf:
        if preview_mode == "Translated (bilingual)":
            render_pdf("Translated Document (Bilingual)", st.session_state.translated_pdf["dual"])
        else:
            render_pdf("Translated Document", st.session_state.translated_pdf["mono"])
    else:
        st.markdown(
            '<div style="border: 2px dashed #d1d5db; border-radius: 12px; padding: 4rem 2rem; text-align: center; color: #9ca3af;">'
            '<p style="font-size: 2rem; margin-bottom: 0.5rem;">🌐</p>'
            '<p>Translated document will appear here</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ─── Download Section ─────────────────────────────────────────────────────────

if st.session_state.translated_pdf:
    st.markdown("#### Download")
    col_dl_mono, col_dl_dual = st.columns(2)
    with col_dl_mono:
        st.download_button(
            label="Download Translated PDF",
            data=st.session_state.translated_pdf["mono"],
            file_name=f"{st.session_state.translated_pdf['filename']}-translated.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col_dl_dual:
        st.download_button(
            label="Download Bilingual PDF",
            data=st.session_state.translated_pdf["dual"],
            file_name=f"{st.session_state.translated_pdf['filename']}-bilingual.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">'
    'DocTranslate &mdash; Professional document translation with full format preservation<br>'
    'Fonts &bull; Tables &bull; Images &bull; Layout &bull; Headers &bull; Footers &bull; Spacing &bull; Alignment'
    '</div>',
    unsafe_allow_html=True,
)
