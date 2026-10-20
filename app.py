"""
VisionCaption AI — Streamlit Application

A polished, fully functional image-captioning web application powered by
the Salesforce BLIP model.  All inference happens locally using the pretrained
weights; no API key is required.
"""

from __future__ import annotations

import datetime
import logging
import textwrap

import streamlit as st
from PIL import Image

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    PAGE_LAYOUT,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    DESCRIPTION_MODES,
    MODE_STANDARD,
    MODE_DETAILED,
    GENERATION_DEFAULTS,
    MODEL_ID,
    MODEL_ARCHITECTURE,
    MODEL_LICENSE,
    MODEL_PAPER,
)
from image_utils import validate_uploaded_file, get_image_info
from model import generate_caption, get_model_info, CaptionResult

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
_CSS = """
<style>
/* ─── Import Google Font ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Root Variables ─── */
:root {
    --primary: #4F46E5;
    --primary-light: #6366F1;
    --primary-lighter: #E0E7FF;
    --primary-dark: #3730A3;
    --surface: #FFFFFF;
    --surface-alt: #F8FAFC;
    --surface-hover: #F1F5F9;
    --text-primary: #1E293B;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --border: #E2E8F0;
    --border-light: #F1F5F9;
    --success: #059669;
    --success-bg: #ECFDF5;
    --warning: #D97706;
    --warning-bg: #FFFBEB;
    --error: #DC2626;
    --error-bg: #FEF2F2;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 25px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
}

/* ─── Global ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ─── Hide Streamlit branding ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ─── Custom card ─── */
.vc-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}
.vc-card:hover {
    box-shadow: var(--shadow-md);
}

/* ─── Header ─── */
.vc-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.vc-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.vc-header p {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ─── Section heading ─── */
.vc-section-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

/* ─── Caption result ─── */
.vc-caption-card {
    background: linear-gradient(135deg, var(--primary-lighter) 0%, #F5F3FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: var(--radius-md);
    padding: 1.75rem;
}
.vc-caption-text {
    font-size: 1.15rem;
    line-height: 1.7;
    color: var(--text-primary);
    font-weight: 500;
}

/* ─── Metadata pill ─── */
.vc-meta {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 0.2rem 0.2rem;
}

/* ─── Steps row ─── */
.vc-step {
    text-align: center;
    padding: 1.25rem 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
}
.vc-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.2rem;
    height: 2.2rem;
    border-radius: 50%;
    background: var(--primary);
    color: #fff;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
}
.vc-step h4 {
    margin: 0.4rem 0 0.25rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
}
.vc-step p {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0;
}

/* ─── Info box (model info / limitations) ─── */
.vc-info-box {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.65;
}

/* ─── Upload area tweaks ─── */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--primary-light) !important;
}

/* ─── Button overrides ─── */
.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.3rem;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: var(--primary-dark) !important;
    box-shadow: var(--shadow-md);
}

/* ─── Divider ─── */
.vc-divider {
    height: 1px;
    background: var(--border);
    margin: 2rem 0;
}

/* ─── Sidebar styling ─── */
section[data-testid="stSidebar"] {
    background: var(--surface-alt);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ─── Empty-state placeholder ─── */
.vc-empty {
    text-align: center;
    padding: 3rem 1.5rem;
    color: var(--text-muted);
}
.vc-empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.vc-empty p {
    font-size: 0.95rem;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ── Session-state defaults ───────────────────────────────────────────────
def _init_state():
    defaults = {
        "caption_result": None,     # CaptionResult | None
        "last_file_id": None,       # to detect new uploads
        "last_mode": MODE_STANDARD,
        "last_max_tokens": GENERATION_DEFAULTS.max_new_tokens,
        "last_num_beams": GENERATION_DEFAULTS.num_beams,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _build_download_text(result: CaptionResult) -> str:
    """Build the plain-text content for the Download Description button."""
    dash = "\u2013"
    lines = [
        "VisionCaption AI \u2014 Generated Description",
        "=" * 44,
        "",
        f"Caption:          {result.caption}",
        f"Mode:             {result.mode}",
        f"Model:            {result.model_id}",
        f"Device:           {result.device.upper()}",
        f"Inference time:   {result.inference_time_seconds:.3f} seconds",
        f"Max new tokens:   {result.generation_params.get('max_new_tokens', dash)}",
        f"Beam count:       {result.generation_params.get('num_beams', dash)}",
        f"Generated at:     {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "---",
        "Note: AI-generated descriptions may contain errors.",
    ]
    return "\n".join(lines)



# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="vc-header">
        <h1>🔬 VisionCaption AI</h1>
        <p>Transform images into meaningful descriptions with AI.
        Upload any image and let a state-of-the-art vision-language model
        describe what it sees — no API key required.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── "How it works" steps ──
cols = st.columns(3)
steps = [
    ("1", "Upload Image", "Select a JPG, PNG, or WEBP image from your device."),
    ("2", "AI Analysis", "The BLIP model processes visual features and generates language."),
    ("3", "Get Description", "Read, copy, or download the generated caption."),
]
for col, (num, title, desc) in zip(cols, steps):
    col.markdown(
        f"""
        <div class="vc-step">
            <div class="vc-step-num">{num}</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="vc-divider"></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR — Generation settings & model info
# ═════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Generation Settings")

    mode = st.selectbox(
        "Description Mode",
        DESCRIPTION_MODES,
        index=0,
        help="Standard gives a concise caption; Detailed attempts a longer description.",
    )

    default_tokens = (
        GENERATION_DEFAULTS.max_new_tokens_detailed
        if mode == MODE_DETAILED
        else GENERATION_DEFAULTS.max_new_tokens
    )
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=GENERATION_DEFAULTS.max_tokens_range[0],
        max_value=GENERATION_DEFAULTS.max_tokens_range[1],
        value=default_tokens,
        step=5,
        help="Upper limit on the number of generated tokens.",
    )

    num_beams = st.slider(
        "Beam Count",
        min_value=GENERATION_DEFAULTS.beam_range[0],
        max_value=GENERATION_DEFAULTS.beam_range[1],
        value=GENERATION_DEFAULTS.num_beams,
        help="Higher beam count may improve quality but increases inference time.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    info = get_model_info()
    st.markdown(
        f"""
        <div class="vc-info-box">
            <strong>Model:</strong> {info['model_id']}<br>
            <strong>Architecture:</strong> {info['architecture']}<br>
            <strong>Framework:</strong> {info['framework']}<br>
            <strong>Device:</strong> {info['device'].upper()}<br>
            <strong>License:</strong> {info['license']}<br>
            <strong>Method:</strong> {info['generation_method']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### ⚠️ Limitations & Privacy")
    st.markdown(
        """
        <div class="vc-info-box">
            <strong>Accuracy:</strong> AI-generated captions may contain
            errors, hallucinations, or miss important details. Do not treat
            them as verified facts.<br><br>
            <strong>Privacy:</strong> Uploaded images are processed in memory
            and are <em>not</em> stored permanently. No data is sent to
            external services; inference runs locally on this machine.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed-mode disclaimer
    if mode == MODE_DETAILED:
        st.info(
            "**Note:** BLIP Base uses a conditional text prompt to encourage "
            "longer output in Detailed mode, but it is not an instruction-following "
            "model. The description may still be relatively short or similar to "
            "the standard caption.",
            icon="ℹ️",
        )


# ═════════════════════════════════════════════════════════════════════════
# MAIN WORKSPACE — Two-column layout
# ═════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1], gap="large")

# ── Left column: Upload & controls ───────────────────────────────────────
with left_col:
    st.markdown('<p class="vc-section-title">📤 Image Upload</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=False,
        label_visibility="collapsed",
        help=f"Supported: {', '.join(ALLOWED_EXTENSIONS).upper()} — max {MAX_UPLOAD_SIZE_MB} MB",
    )

    image: Image.Image | None = None

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        img, error = validate_uploaded_file(uploaded.name, file_bytes)

        if error:
            st.error(f"❌ {error}", icon="🚫")
        else:
            image = img
            info_dict = get_image_info(image)

            st.image(image, caption=uploaded.name, use_container_width=True)

            # Metadata pills
            st.markdown(
                f"""
                <div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.4rem;">
                    <span class="vc-meta">📄 {uploaded.name}</span>
                    <span class="vc-meta">📐 {info_dict['width']} × {info_dict['height']} px</span>
                    <span class="vc-meta">🎨 {info_dict['mode']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")  # spacer

    # ── Generate / Regenerate button ──
    # Determine file identity for change detection
    current_file_id = None
    if uploaded is not None and image is not None:
        current_file_id = f"{uploaded.name}_{uploaded.size}"

    settings_changed = (
        mode != st.session_state.last_mode
        or max_tokens != st.session_state.last_max_tokens
        or num_beams != st.session_state.last_num_beams
    )
    file_changed = current_file_id != st.session_state.last_file_id

    # Button label
    has_previous = st.session_state.caption_result is not None
    btn_label = "🔄 Regenerate Description" if (has_previous and not file_changed) else "✨ Generate Description"

    generate_pressed = st.button(
        btn_label,
        type="primary",
        disabled=(image is None),
        use_container_width=True,
    )

    # Should we run inference?
    run_inference = False
    if generate_pressed and image is not None:
        run_inference = True
    elif image is not None and file_changed and not has_previous:
        # First upload — auto-generate
        pass  # User must click the button

    if run_inference:
        st.session_state.last_file_id = current_file_id
        st.session_state.last_mode = mode
        st.session_state.last_max_tokens = max_tokens
        st.session_state.last_num_beams = num_beams


# ── Right column: Results ────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="vc-section-title">📝 Generated Description</p>', unsafe_allow_html=True)

    if run_inference and image is not None:
        with st.spinner("🧠 Analyzing image — this may take a moment on first run…"):
            result = generate_caption(
                image,
                mode=mode,
                max_new_tokens=max_tokens,
                num_beams=num_beams,
            )
            st.session_state.caption_result = result

    result: CaptionResult | None = st.session_state.caption_result

    if result is not None and result.success:
        # Caption card
        st.markdown(
            f"""
            <div class="vc-caption-card">
                <p class="vc-caption-text">{result.caption}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metadata
        st.markdown(
            f"""
            <div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin:0.75rem 0;">
                <span class="vc-meta">🤖 {result.model_id}</span>
                <span class="vc-meta">🔧 {result.mode}</span>
                <span class="vc-meta">⏱️ {result.inference_time_seconds:.2f}s</span>
                <span class="vc-meta">💻 {result.device.upper()}</span>
                <span class="vc-meta">🔢 Beams: {result.generation_params.get('num_beams', '–')}</span>
                <span class="vc-meta">📏 Max tokens: {result.generation_params.get('max_new_tokens', '–')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actions
        act_cols = st.columns([1, 1])

        # Copy-friendly text (Streamlit has st.code for easy copy)
        with act_cols[0]:
            st.code(result.caption, language=None)

        # Download
        with act_cols[1]:
            download_content = _build_download_text(result)
            st.download_button(
                label="⬇️ Download Description",
                data=download_content,
                file_name=f"visioncaption_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    elif result is not None and not result.success:
        st.error(f"❌ {result.error_message}", icon="⚠️")

    else:
        # Empty state
        st.markdown(
            """
            <div class="vc-empty">
                <div class="vc-empty-icon">🖼️</div>
                <p>Upload an image and click <strong>Generate Description</strong><br>
                to see the AI-generated caption here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════
st.markdown('<div class="vc-divider"></div>', unsafe_allow_html=True)

footer_cols = st.columns([2, 1])
with footer_cols[0]:
    st.markdown(
        f"""
        <div class="vc-info-box">
            <strong>About VisionCaption AI</strong><br>
            This application uses the <em>{MODEL_ARCHITECTURE}</em> model
            (<code>{MODEL_ID}</code>) to generate natural-language descriptions
            of uploaded images.  The model was pretrained on large-scale
            image-text pairs and is used here for inference only — no training
            or fine-tuning is performed at runtime.<br><br>
            <strong>Reference:</strong> {MODEL_PAPER}<br>
            <strong>License:</strong> {MODEL_LICENSE}
        </div>
        """,
        unsafe_allow_html=True,
    )
with footer_cols[1]:
    st.markdown(
        """
        <div class="vc-info-box" style="height:100%;">
            <strong>Disclaimer</strong><br>
            Generated descriptions may be inaccurate, incomplete, or
            misleading.  They should not be treated as verified factual
            statements.  This tool is intended for educational and
            demonstration purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )


