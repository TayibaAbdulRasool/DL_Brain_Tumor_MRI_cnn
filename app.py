"""
NeuroScan AI — Brain Tumor MRI Classifier
A Streamlit app that loads a trained CNN (.pth) and classifies
brain MRI scans into: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
"""

import io
import time

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MODEL_PATH = "brain_tumor_mri_cnn.pth"
IMG_SIZE = 128
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_DESCRIPTIONS = {
    "Glioma": "A tumor that originates in the glial cells of the brain or spine.",
    "Meningioma": "A tumor arising from the meninges, the membranes surrounding the brain and spinal cord.",
    "No Tumor": "No tumor detected in the scan.",
    "Pituitary": "A tumor located in the pituitary gland at the base of the brain.",
}
# Single professional accent (deep blue/cyan) for data, muted amber reserved for
# attention states, muted green for the "normal" / no-tumor state.
CLASS_ACCENT = {
    "Glioma": "#38BDF8",
    "Meningioma": "#38BDF8",
    "No Tumor": "#34D399",
    "Pituitary": "#38BDF8",
}

# ------------------------------------------------------------------
# Model definition (matches the state_dict shapes in the .pth file)
# conv1: 3->32, conv2: 32->64, conv3: 64->128, each + ReLU + MaxPool(2)
# 128px input -> 16px feature map -> flatten 128*16*16 = 32768
# fc1: 32768->128, fc2: 128->4
# ------------------------------------------------------------------
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@st.cache_resource(show_spinner=False)
def load_model():
    model = BrainTumorCNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict(model, tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    return probs


# ------------------------------------------------------------------
# Styling — clinical / SaaS dashboard aesthetic
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
    --bg-primary: #071412;
    --bg-secondary: #0D1F1B;
    --bg-elevated: #132A24;

    --border-subtle: #214139;
    --border-strong: #31594C;

    --text-primary: #F1F7F5;
    --text-secondary: #A8BCB5;
    --text-muted: #718C83;

    --accent: #2DD4A8;
    --accent-hover: #5EEAD4;
    --accent-soft: rgba(45, 212, 168, 0.12);

    --status-warn: #F6C453;
    --status-warn-soft: rgba(246, 196, 83, 0.12);

    --status-ok: #4ADE80;
    --status-ok-soft: rgba(74, 222, 128, 0.12);
}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background: var(--bg-primary);
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: var(--text-primary);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.25rem;
    }

    .brand-block {
        padding: 0 0.25rem 1.25rem 0.25rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.25rem;
    }
    .brand-mark {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.15rem;
    }
    .brand-mark-dot {
        width: 9px;
        height: 9px;
        border-radius: 2px;
        background: var(--accent);
        flex-shrink: 0;
    }
    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-primary);
    }
    .brand-tagline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--accent);
        margin-left: 1.4rem;
        letter-spacing: 0.02em;
    }

    .side-section {
        margin-bottom: 1.5rem;
    }
    .side-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.55rem;
    }
    .side-fact-row {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        padding: 0.28rem 0;
        color: var(--text-secondary);
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .side-fact-row span:last-child {
        color: var(--text-primary);
        font-weight: 500;
    }

    .class-chip {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.82rem;
        color: var(--text-secondary);
        padding: 0.3rem 0;
    }
    .class-chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .side-disclaimer {
        font-size: 0.74rem;
        line-height: 1.5;
        color: var(--text-muted);
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
    }
    .side-disclaimer strong {
        display: block;
        color: var(--text-secondary);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }

    /* ---------------- Page header ---------------- */
    .page-header {
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .page-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.92rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* ---------------- Panels ---------------- */
    .panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
    }
    .panel-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--text-muted);
        margin-bottom: 1.1rem;
    }

    /* Upload dropzone */
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--bg-elevated);
        border: 1px dashed var(--border-strong);
        border-radius: 8px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
    }
    section[data-testid="stFileUploaderDropzoneInstructions"] span,
    section[data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--text-secondary) !important;
    }

    /* Uploaded image frame */
    .image-frame {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 0.9rem;
        margin-top: 1rem;
        text-align: center;
    }
    .image-frame-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.6rem;
        letter-spacing: 0.02em;
    }
    .image-frame img {
        border-radius: 4px;
    }

    /* Buttons */
    /* Professional Emerald Button */
.stButton > button {
    background: #2DD4A8;
    color: #071412;
    border: 1px solid #2DD4A8;
    border-radius: 7px;
    padding: 0.65rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(45, 212, 168, 0.12);
}

.stButton > button:hover {
    background: #5EEAD4;
    border-color: #5EEAD4;
    color: #071412;
    box-shadow: 0 4px 14px rgba(45, 212, 168, 0.20);
    transform: translateY(-1px);
}

.stButton > button:active {
    background: #2DD4A8;
    transform: translateY(0);
}

    /* ---------------- Results ---------------- */
    .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.1rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .result-class {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .confidence-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--accent);
        text-align: right;
    }
    .result-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1.3rem;
        line-height: 1.5;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        margin-bottom: 1.3rem;
    }
    .status-badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }
    .status-ok {
        background: var(--status-ok-soft);
        color: var(--status-ok);
        border: 1px solid rgba(52, 178, 122, 0.25);
    }
    .status-ok .status-badge-dot { background: var(--status-ok); }
    .status-warn {
        background: var(--status-warn-soft);
        color: var(--status-warn);
        border: 1px solid rgba(217, 164, 65, 0.25);
    }
    .status-warn .status-badge-dot { background: var(--status-warn); }

    .prob-section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.85rem;
    }
    .prob-row {
        margin-bottom: 0.7rem;
    }
    .prob-label-row {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-bottom: 0.3rem;
    }
    .prob-label-row .prob-name { color: var(--text-primary); font-weight: 500; }
    .prob-track {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        background: var(--accent);
    }

    .empty-state {
        font-size: 0.85rem;
        color: var(--text-muted);
        padding: 2.5rem 0.5rem;
        text-align: center;
        line-height: 1.6;
    }

    .inline-disclaimer {
        font-size: 0.76rem;
        line-height: 1.55;
        color: var(--text-muted);
        border-top: 1px solid var(--border-subtle);
        padding-top: 0.9rem;
        margin-top: 1.3rem;
    }

    .footer-note {
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.72rem;
        margin-top: 2.5rem;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-block">
            <div class="brand-mark">
                <div class="brand-mark-dot"></div>
                <div class="brand-name">NeuroScan AI</div>
            </div>
            <div class="brand-tagline">MRI Brain Tumor Screening</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_chip_html = "".join(
        f"""<div class="class-chip">
                <div class="class-chip-dot" style="background:{CLASS_ACCENT[name]};"></div>
                <span>{name}</span>
            </div>"""
        for name in CLASS_NAMES
    )
    st.markdown(
        f"""
        <div class="side-section">
            <div class="side-section-title">Detectable Classes</div>
            {class_chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-disclaimer">
            <strong>Educational Use Only</strong>
            This tool is a research and educational demonstration. It is not a
            certified diagnostic device and must not be used to guide clinical
            decisions. Always consult a licensed radiologist or physician.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="page-header">
        <div class="page-eyebrow">Diagnostic Imaging</div>
        <div class="page-title">Brain MRI Analysis</div>
        <div class="page-subtitle">AI-assisted classification of brain MRI scans</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model weights from `{MODEL_PATH}`: {e}")

# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Upload Scan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Accepted formats: JPG, JPEG, PNG</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload MRI scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    analyze_clicked = False
    image = None

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="image-frame-caption">{uploaded_file.name}</div></div>',
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button("Analyze Scan", use_container_width=True)
    else:
        st.markdown(
            '<div class="empty-state">No image uploaded yet.<br>Drag and drop a file above, or browse to select one.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Analysis Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Predicted class, confidence, and full probability distribution</div>',
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.warning("Model is not loaded. Please check the weights file path.")
    elif uploaded_file is None:
        st.markdown(
            '<div class="empty-state">Results will appear here once a scan has been uploaded and analyzed.</div>',
            unsafe_allow_html=True,
        )
    elif analyze_clicked:
        with st.spinner("Running inference..."):
            tensor = preprocess_image(image)
            time.sleep(0.4)
            probs = predict(model, tensor)

        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx]) * 100

        st.markdown(
            f"""
            <div class="result-top">
                <div>
                    <div class="result-label">Predicted Class</div>
                    <div class="result-class">{top_class}</div>
                </div>
                <div>
                    <div class="result-label" style="text-align:right;">Confidence</div>
                    <div class="confidence-value">{top_conf:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-desc">{CLASS_DESCRIPTIONS[top_class]}</div>',
            unsafe_allow_html=True,
        )

        if top_class == "No Tumor":
            st.markdown(
                """
                <div class="status-badge status-ok">
                    <div class="status-badge-dot"></div>
                    No abnormal indicators detected
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-badge status-warn">
                    <div class="status-badge-dot"></div>
                    Indicators consistent with {top_class} — clinical review recommended
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="prob-section-title">Probability Distribution</div>', unsafe_allow_html=True)
        order = np.argsort(probs)[::-1]
        for idx in order:
            name = CLASS_NAMES[idx]
            pct = float(probs[idx]) * 100
            st.markdown(
                f"""
                <div class="prob-row">
                    <div class="prob-label-row">
                        <span class="prob-name">{name}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="inline-disclaimer">
                This prediction is generated by an AI model and may be inaccurate.
                Always confirm findings with a licensed medical professional before
                making any healthcare decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="empty-state">Click <b>Analyze Scan</b> to run the model on the uploaded image.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">NeuroScan AI &middot; PyTorch + Streamlit &middot; For research and educational use only</div>',
    unsafe_allow_html=True,
)
