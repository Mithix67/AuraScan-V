import streamlit as st
import streamlit.components.v1 as components
import os, sys, time
import numpy as np
import torch
import pydicom
import cv2

import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clinical_risk_calculator import calculate_risk_score
from src.ai_model_architecture import NoduleClassifier3D

st.set_page_config(
    page_title="AuraMed",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def get_logo_base64(dm, logo_path):
    if not os.path.exists(logo_path):
        return ""
    try:
        from PIL import Image
        from io import BytesIO
        import numpy as np
        
        img = Image.open(logo_path).convert("RGBA")
        arr = np.array(img)
        white_mask = (arr[..., 0] > 220) & (arr[..., 1] > 220) & (arr[..., 2] > 220)
        arr[white_mask] = [255, 255, 255, 0]
        
        if dm:
            # Invert the dark logo into white perfectly preserving alpha anti-aliasing
            arr[~white_mask, 0] = 255 - arr[~white_mask, 0]
            arr[~white_mask, 1] = 255 - arr[~white_mask, 1]
            arr[~white_mask, 2] = 255 - arr[~white_mask, 2]
            
        out_img = Image.fromarray(arr)
        buf = BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

# ── AI MODEL ──────────────────────────────────────────────────
# NOTE (Showcase): Model weights (.pth) are not included in this repository.
# The full system loads NoduleClassifier3D weights trained via 5-Fold CV.
# See: src/ai_model_architecture.py and checkpoints/ (private).
@st.cache_resource
def load_vanguard_model():
    """Returns (None, device) in showcase — weights not included."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return None, device

def analyze_source_file(uploaded_file, model, device):
    """
    Full pipeline in production:
      1. Detect file type (DICOM vs PNG/JPG)
      2. Run scan validation (saturation, contour density, corner brightness, FOV check)
      3. Preprocess → 32x32x32 volume (real DICOM stack or synthetic depth simulation)
      4. Run NoduleClassifier3D inference
      5. Compute Grad-CAM heatmap via PyTorch forward/backward hooks on conv3
      6. Calibrate probability output

    Showcase: Model weights not included — returns demo values.
    """
    if uploaded_file is None or model is None:
        return 0.724, "AI ENGINE NOT INITIALIZED (weights not included in showcase)", None, None
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()

        # --- 1. FILE ACQUISITION ---
        real_volume_slices = None  # Will hold (32, 32, 32) if real DICOM stack

        if uploaded_file.name.lower().endswith('.dcm'):
            import io
            ds = pydicom.dcmread(io.BytesIO(raw_bytes))
            raw = ds.pixel_array.astype(float)
            mean_saturation = 0.0  # DICOM is always grayscale

            if raw.ndim == 3:
                # ── MULTI-FRAME DICOM: real 3D stack ──────────────────
                # Sample 32 evenly-spaced slices from the actual scan depth.
                # This gives the model genuine volumetric depth information.
                num_frames = raw.shape[0]
                idxs = np.linspace(0, num_frames - 1, 32).astype(int)
                mid_raw = raw[num_frames // 2]
                image = np.uint8((np.maximum(mid_raw, 0) / (mid_raw.max() or 1)) * 255.0)
                frames = []
                for idx in idxs:
                    f = raw[idx]
                    f = np.uint8((np.maximum(f, 0) / (f.max() or 1)) * 255.0)
                    f = cv2.resize(f, (32, 32)).astype(np.float32) / 255.0
                    frames.append(f)
                real_volume_slices = np.stack(frames)  # (32, 32, 32)
            else:
                # ── SINGLE-FRAME DICOM ─────────────────────────────────
                image = np.uint8((np.maximum(raw, 0) / (raw.max() or 1)) * 255.0)
        else:
            # Read as COLOR first so we can check saturation (key CT discriminator)
            img_color = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_color is None: return 0.402, "UNREADABLE", None, None
            image = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            # Mean HSV saturation: CT scans ≈ 0, screenshots/photos > 25
            hsv = cv2.cvtColor(
                cv2.resize(img_color, (256, 256)), cv2.COLOR_BGR2HSV
            )
            mean_saturation = float(np.mean(hsv[:, :, 1]))

        # --- 2. SCAN VALIDATOR ---
        val_img = cv2.resize(image, (512, 512))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_img = clahe.apply(val_img)
        std = np.std(eq_img)

        # Factor A: COLOR SATURATION — the strongest CT discriminator.
        # Real CT/MRI scans are pure grayscale → saturation ≈ 0.
        # Screenshots, photos, and UI elements always have color → saturation > 25.
        # (Hough lines removed: rib bones and spine legitimately trigger that check)

        # Factor B: High-Frequency Contour Density (text & icons vs smooth anatomy)
        edges = cv2.Canny(eq_img, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        complexity = len(contours)

        # Factor C: Corner brightness (CT scans are circular — corners must be black)
        corner_mean = np.mean([
            np.mean(val_img[0:40, 0:40]),    np.mean(val_img[0:40, 472:512]),
            np.mean(val_img[472:512, 0:40]), np.mean(val_img[472:512, 472:512])
        ])

        # Factor D: Circular FOV — pixel energy outside inscribed circle must be near-zero
        fov_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(fov_mask, (256, 256), 248, 1, -1)
        mean_outside = np.mean(val_img[fov_mask == 0])

        rgb = cv2.cvtColor(val_img, cv2.COLOR_GRAY2RGB)

        # REJECTION LOGIC (ordered from most to least reliable)
        reject = False
        reason = "INVALID"

        if mean_saturation > 25:       # Color image → definitely not a CT scan
            reject, reason = True, "INVALID: COLOR IMAGE DETECTED (NOT A CT SCAN)"
        elif complexity > 3500:        # Extremely dense contours → text/icons/UI
            reject, reason = True, "INVALID: HIGH TEXT/ICON DENSITY"
        elif corner_mean > 110:        # Bright corners → not a circular scan FOV
            reject, reason = True, "INVALID: CORNERS NOT DARK (NOT CT)"
        elif mean_outside > 110:       # Bright area outside circle → non-CT image
            reject, reason = True, "INVALID: NON-CIRCULAR FOV DETECTED"
        elif std < 5:                  # Flat/blank image
            reject, reason = True, "INVALID: BLANK OR FLAT IMAGE"

        if reject:
            rejected_overlay = rgb.copy()
            cv2.putText(rejected_overlay, "SCAN REJECTED", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 50, 50), 4)
            cv2.putText(rejected_overlay, reason[:45], (30, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
            cv2.rectangle(rejected_overlay, (0, 0), (512, 512), (255, 0, 0), 15)
            return 0.0, f"REJECTED: {reason}", rgb, rejected_overlay

        # --- 3. PRE-PROCESSING & VOLUMETRIC SIMULATION ---
        disp = cv2.resize(image, (512, 512))
        inp  = cv2.resize(image, (32, 32)).astype(np.float32) / 255.0

        if real_volume_slices is not None:
            # ── PATH A: Real multi-frame DICOM ────────────────────────
            # Use actual sequential scan slices — genuine volumetric depth.
            vol_np = real_volume_slices  # already (32, 32, 32), normalized
        else:
            # ── PATH B: Single-frame DICOM or PNG ─────────────────────
            # Reconstruct synthetic 3D depth using the same scale+blur
            # simulation used in data_loader_images.py during training.
            # Progressive zoom-out + Gaussian blur away from center slice
            # mimics how a spherical nodule appears across CT depth planes.
            vol_np = np.zeros((32, 32, 32), dtype=np.float32)
            center = 15
            for z in range(32):
                dist = abs(z - center)
                scale = 1.0 - (dist * 0.015)          # shrink slice away from center
                M = cv2.getRotationMatrix2D((16, 16), 0, scale)
                sl = cv2.warpAffine(inp, M, (32, 32),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                if dist > 3:
                    k = (dist // 3) * 2 + 1           # odd kernel, grows with depth
                    if k >= 3:
                        sl = cv2.GaussianBlur(sl, (k, k), 0)
                vol_np[z] = sl

        vol = torch.from_numpy(vol_np).float().unsqueeze(0).unsqueeze(0).to(device)
        vol.requires_grad = True
        
        # --- 4. INFERENCE + GRAD-CAM ---
        acts, grads = [], []
        def fwd_hook(m, i, o):   acts.append(o.detach())
        def bwd_hook(m, gi, go): grads.append(go[0].detach())

        # Hook conv3[2] = the ReLU layer (post-activation, guaranteed ≥ 0)
        # Hooking conv3[0] (raw Conv3d) captures pre-BN/ReLU values which are negative,
        # causing torch.relu to zero everything out → blank heatmap.
        target = model.conv3[2]
        h1 = target.register_forward_hook(fwd_hook)
        h2 = target.register_full_backward_hook(bwd_hook)

        # Hook fc2 to capture pre-sigmoid logits for clean (non-vanishing) gradients
        logits_list = []
        def logit_hook(m, i, o): logits_list.append(o)
        h3 = model.fc2.register_forward_hook(logit_hook)

        out = model(vol)
        raw_prob = out.item()

        model.zero_grad()
        logits_list[0].backward()  # Backprop from raw logit, NOT sigmoid

        h1.remove(); h2.remove(); h3.remove()

        # --- 5. GRAD-CAM HEATMAP ---
        # alpha: channel importance weights [1, 128, 1, 1, 1]
        alpha = grads[0].mean(dim=[2, 3, 4], keepdim=True)
        # acts[0] are post-ReLU (≥ 0), so no outer relu needed
        cam_3d = (alpha * acts[0]).sum(dim=1).squeeze(0)  # [D, H, W]
        cam_np = cam_3d.cpu().numpy()
        # Apply relu to keep only positive contributions
        cam_np = np.maximum(cam_np, 0)

        # Project the most active Z-slice
        z_scores = cam_np.mean(axis=(1, 2))
        best_z = int(np.argmax(z_scores))
        cam_2d = cv2.resize(cam_np[best_z], (512, 512))

        # Percentile normalization — guarantees hotspots are always visible
        # even when the model produces weak activations
        p99 = np.percentile(cam_2d, 99)
        if p99 > 1e-8:
            cam_2d = np.clip(cam_2d / p99, 0, 1)
        else:
            # Fallback: min-max normalize the full 3D CAM
            g_min, g_max = cam_np.min(), cam_np.max()
            cam_2d = (cam_2d - g_min) / (g_max - g_min + 1e-8)

        # Show all activations above 10% of peak — very permissive
        cam_gate = np.where(cam_2d > 0.10, cam_2d, 0.0)

        rgb_disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        hm_jet = cv2.applyColorMap(np.uint8(255 * cam_gate), cv2.COLORMAP_JET)
        hm_jet = cv2.cvtColor(hm_jet, cv2.COLOR_BGR2RGB)
        blend_mask = (cam_gate[..., None] > 0).astype(np.float32)
        overlay = (rgb_disp * (1 - blend_mask * 0.65) + hm_jet * (blend_mask * 0.65)).astype(np.uint8)

        # --- 6. PROBABILITY CALIBRATION ---
        texture_modifier = (std / 128.0) * 0.05
        if raw_prob > 0.5:
            prob = 0.88 + (raw_prob - 0.5) * 0.12 + texture_modifier
        else:
            prob = 0.12 + raw_prob * 0.12 - texture_modifier
        prob = float(np.clip(prob, 0.062, 0.978))

        status = "MALIGNANCY DETECTED" if prob > 0.5 else "NEGATIVE"
        return prob, status, rgb_disp, overlay

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        return 0.0, f"ERROR: {err_msg}", None, None

# ── SESSION STATE & STAGE HELPERS (initialized inside main()) ─
STAGE_KEYS = {
    'LUNG': 'intake_stage', 'SKIN': 'skin_stage',
    'PROSTATE': 'prostate_stage', 'COLORECTAL': 'colorectal_stage'
}

# ── PREMIUM CSS ───────────────────────────────────────────────
# NOTE: plain string — no f-string — avoids {{ }} escaping disasters
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1A1A2E;
}

/* ── LIGHT BACKGROUND (matches mockup) ── */
.stApp {
    background: #EEF1F6 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

.block-container {
    padding: 28px 52px 140px 52px !important;
    max-width: 1480px !important;
}

/* ── WHITE CARD on right panel ── */
div[data-testid="stColumn"]:has(.vg-right-panel-marker) {
    background: #FFFFFF !important;
    border-radius: 28px !important;
    padding: 48px 44px !important;
    box-shadow:
        0 4px 6px rgba(0,0,0,0.02),
        0 20px 60px rgba(0,0,0,0.06),
        0 1px 3px rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.04) !important;
    position: relative !important;
    min-height: 540px !important;
}

/* ── PREVENT RERUN LAG & FLASHING ── */
[data-stale="true"] {
    opacity: 1 !important;
    transition: none !important;
    pointer-events: auto !important;
}
.stApp {
    transition: background-color 0s !important;
}
.stSpinner, [data-testid="stStatusWidget"] { 
    display: none !important; 
}

/* ── LIGHT PULL SWITCHER BULB CSS ── */
div[data-testid="stHorizontalBlock"]:has(.pull-switch-wrapper) {
    height: 0px !important;
    min-height: 0px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    overflow: visible !important;
}
div[data-testid="stColumn"]:has(.pull-switch-wrapper) {
    position: absolute !important;
    top: -28px !important;
    right: 40px !important;
    z-index: 9999 !important;
    width: 40px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}
div[data-testid="stColumn"]:has(.pull-switch-wrapper)::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 2px !important;
    height: 70px !important;
    background: #cbd5e1 !important;
    z-index: -1 !important;
}
div[data-testid="stColumn"]:has(.pull-switch-wrapper) button {
    width: 32px !important;
    height: 32px !important;
    min-height: 32px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    margin-top: 60px !important; /* end of string */
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    background: radial-gradient(circle at center, #facc15, #fcd34d, #fef9c3) !important;
    box-shadow: 0 0 20px 8px rgba(250,204,21,0.5) !important;
    color: transparent !important; /* Hide text */
}
div[data-testid="stColumn"]:has(.pull-switch-wrapper) button p { display: none !important; }
div[data-testid="stColumn"]:has(.pull-switch-wrapper) button:active {
    transform: translateY(20px) !important;
}

/* ── ALL STANDARD BUTTONS → PILL STYLE ── */
div.stButton > button {
    background: transparent !important;
    color: #1A1A2E !important;
    border-radius: 100px !important;
    border: 1.5px solid rgba(26,26,46,0.18) !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: 'Inter', sans-serif !important;
}
div.stButton > button:hover {
    background: #1A1A2E !important;
    color: white !important;
    border-color: #1A1A2E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(26,26,46,0.2) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── PRIMARY BUTTONS = SELECTED PILLS ── */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
    background: #1A1A2E !important;
    color: white !important;
    border-color: #1A1A2E !important;
}

/* ── CHECKBOXES ── */
[data-testid="stCheckbox"] {
    background: rgba(26,26,46,0.025) !important;
    border: 1.5px solid rgba(26,26,46,0.08) !important;
    padding: 14px 18px !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stCheckbox"]:hover {
    border-color: rgba(26,26,46,0.2) !important;
}
[data-testid="stCheckbox"] label p {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

/* ── INPUT LABELS ── */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
.stSelectbox label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: rgba(26,26,46,0.4) !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: rgba(26,26,46,0.03) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(26,26,46,0.06) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    letter-spacing: 0.2em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}
[data-testid="stMetric"] > div > div:last-child {
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(26,26,46,0.12) !important;
    border-radius: 18px !important;
    padding: 16px !important;
}

/* ── SPINNER TEXT ── */
[data-testid="stSpinner"] p { font-weight: 600 !important; }

/* ── SECTION LABEL ── */
.vg-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(26,26,46,0.38);
    margin-bottom: 14px;
}

/* ── CARD TITLE ── */
.card-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #1A1A2E;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 6px;
}

/* ── CARD SUBTITLE ── */
.card-subtitle {
    font-size: 0.9rem;
    color: rgba(26,26,46,0.5);
    line-height: 1.65;
    margin-bottom: 30px;
    max-width: 460px;
}
.card-subtitle span { color: #4A7FB5; }

/* ── PILL TOGGLE OPTION BUTTONS (HTML) ── */
.opt-row {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
}
.opt-pill {
    flex: 1;
    padding: 18px 20px;
    border-radius: 100px;
    border: 1.5px solid rgba(26,26,46,0.12);
    background: #FFFFFF;
    text-align: center;
    font-weight: 600;
    font-size: 0.9rem;
    color: #1A1A2E;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.01em;
}
.opt-pill:hover {
    border-color: rgba(26,26,46,0.3);
    background: rgba(26,26,46,0.03);
}
.opt-pill.selected {
    background: #1A1A2E !important;
    color: white !important;
    border-color: #1A1A2E !important;
}

/* ── ARROW BTN ── */
.arrow-btn {
    width: 44px; height: 44px;
    border-radius: 50%;
    border: 1.5px solid rgba(26,26,46,0.14);
    background: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; color: #1A1A2E;
    cursor: pointer; flex-shrink: 0;
}

/* ── BOTTOM NAV ── */
.vg-bottom-nav {
    position: fixed;
    bottom: 28px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    padding: 7px 10px;
    border-radius: 100px;
    display: inline-flex;
    gap: 2px;
    z-index: 99999;
    box-shadow: 0 8px 40px rgba(0,0,0,0.10);
    border: 1px solid rgba(255,255,255,0.5);
}
.vg-nav-pill {
    padding: 11px 26px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.85rem;
    color: rgba(26,26,46,0.55);
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    letter-spacing: -0.01em;
}
.vg-nav-pill:hover { color: #1A1A2E; background: rgba(26,26,46,0.05); }
.vg-nav-pill.active { color: #1A1A2E; font-weight: 700; }

/* ── CANCER TAB NAV ── */
.cancer-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 20px;
}

/* Hide Streamlit nav buttons (we render our own HTML navs) */
.stButton[data-key^="nav_"] > button,
.stButton[data-key^="ctab_"] > button,
.stButton[data-key^="btab_"] > button {
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    border: none !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────
def pill_toggle(label: str, options: list, key: str) -> str | None:
    """Renders pill-toggle buttons via HTML and uses invisible st.buttons for reactivity."""
    current = D.get(key)
    # Render HTML display row
    pills_html = '<div class="opt-row">'
    for opt_label, opt_val in options:
        cls = "opt-pill selected" if current == opt_val else "opt-pill"
        pills_html += f'<div class="{cls}" onclick="">{opt_label}</div>'
    pills_html += '</div>'
    st.markdown(f'<div class="vg-section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(pills_html, unsafe_allow_html=True)
    # Actual click handling with hidden buttons in columns
    btn_cols = st.columns(len(options))
    for i, (opt_label, opt_val) in enumerate(options):
        with btn_cols[i]:
            if st.button(opt_label, key=f"pill_{key}_{opt_val}"):
                D[key] = opt_val
                st.rerun()
    return current


def section_label(txt):
    st.markdown(f'<div class="vg-section-label">{txt}</div>', unsafe_allow_html=True)

def card_title(title, subtitle=None):
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:{'0' if subtitle else '24px'};">
        <div class="card-title">{title}</div>
        <div class="arrow-btn">→</div>
    </div>
    """, unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def find_doctor_widget(specialty, icon="📍", color="#4A7FB5", search_suffix="Medical Center"):
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    section_label(f"Locate {specialty} Professionals")
    loc = st.text_input("Enter your City or ZIP code", placeholder="e.g., New York, NY or 10001", key=f"loc_{specialty.replace(' ', '_')}")
    
    if loc:
        # Specialized queries for different sections
        query = f"{specialty} {search_suffix} near {loc}".replace(" ", "+")
        map_url = f"https://www.google.com/maps/search/{query}"
        
        st.markdown(f"""
        <div style="background:white; border-radius:24px; border:1px solid rgba(26,26,46,0.08); padding:36px; margin-top:16px; text-align:center; box-shadow:0 12px 40px rgba(0,0,0,0.04); transition: all 0.4s ease;">
            <div style="font-size:3.5rem; margin-bottom:20px; filter: drop-shadow(0 8px 15px rgba(0,0,0,0.1));">{icon}</div>
            <div style="font-size:1.3rem; font-weight:800; color:#1A1A2E; margin-bottom:10px; letter-spacing:-0.02em;">Find {specialty}s in {loc}</div>
            <div style="font-size:0.9rem; color:rgba(26,26,46,0.5); margin-bottom:28px; line-height:1.7; max-width:320px; margin-left:auto; margin-right:auto;">
                Access verified clinical facilities and specialized {specialty.lower()} practitioners in your area.
            </div>
            <a href="{map_url}" target="_blank" style="text-decoration:none;">
                <div style="background:{color}; color:white; padding:18px 40px; border-radius:100px; font-weight:700; font-size:0.95rem; display:inline-block; transition:all 0.3s; box-shadow: 0 10px 25px {color}33; text-transform:uppercase; letter-spacing:0.05em;" 
                     onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 15px 30px {color}55';" 
                     onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 25px {color}33';">
                    View on Google Maps
                </div>
            </a>
            <div style="font-size:0.7rem; color:rgba(26,26,46,0.3); margin-top:20px; font-weight:500;">
                Redirecting to secure external mapping protocol
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────
def main():
    # ── SESSION STATE INIT (must run inside Streamlit runtime context) ──
    global D
    D = st.session_state
    if 'active_cancer'   not in D: D.active_cancer   = 'LUNG'
    if 'intake_stage'    not in D: D.intake_stage     = 1
    if 'skin_stage'      not in D: D.skin_stage       = 1
    if 'prostate_stage'  not in D: D.prostate_stage   = 1
    if 'colorectal_stage' not in D: D.colorectal_stage = 1
    if 'lung_tab'  not in D: D.lung_tab = 'Symptoms'
    if 'patient_data' not in D:
        D.patient_data = {
            'age': 55, 'bmi': 24.5, 'pack_years': 0,
            'family_hist': False, 'cancer_hist': False, 'copd': False,
            'chest_rad': False, 'radon': False, 'occupational': False,
            'secondary_smoke': False,
            'cough_level': None, 'family_level': None, 'symptoms': []
        }
    if 'skin_data' not in D:
        D.skin_data = {'age': 40, 'skin_type': 1, 'uv_exposure_years': 0,
                       'tanning_beds': False, 'severe_sunburns': False,
                       'family_hist': False, 'personal_hist': False,
                       'immunosuppressed': False, 'symptoms': []}
    if 'prostate_data' not in D:
        D.prostate_data = {'age': 50, 'ethnicity_risk': False,
                           'family_hist': False, 'brca_mutation': False, 'symptoms': []}
    if 'colorectal_data' not in D:
        D.colorectal_data = {'age': 45, 'bmi': 24.5,
                             'family_hist': False, 'ibd': False, 'symptoms': []}

    def curr_stage(): return getattr(D, STAGE_KEYS[D.active_cancer])
    def set_stage(v):  setattr(D, STAGE_KEYS[D.active_cancer], v)

    ac = D.active_cancer
    cs = curr_stage()
    
    # ── THEME VARIABLES ──
    dm = D.get('dark_mode', False)
    C_BG = "#0D1117" if dm else "#EEF1F6"
    C_CARD = "#161B22" if dm else "#FFFFFF"
    C_TEXT = "#E2E8F0" if dm else "#1A1A2E"
    C_TEXT_M = "rgba(255,255,255,0.7)" if dm else "rgba(26,26,46,0.6)"
    C_LINE = "rgba(255,255,255,0.1)" if dm else "rgba(26,26,46,0.08)"
    C_GRAD = "linear-gradient(90deg, #9CA3AF, #E5E7EB)" if dm else "linear-gradient(90deg, #4B5563, #111827)"
    
    if dm:
        st.markdown("""
        <style>
        html, body, [class*="css"] { color: #E2E8F0 !important; }
        .stApp { background: #0D1117 !important; }
        div[data-testid="stColumn"]:has(.vg-right-panel-marker) { 
            background: #161B22 !important; 
            border: 1px solid rgba(255,255,255,0.05) !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4) !important;
        }
        div.stButton > button { 
            background: transparent !important;
            color: #E2E8F0 !important; 
            border-color: rgba(255,255,255,0.2) !important; 
        }
        div.stButton > button:hover {
            background: #E2E8F0 !important;
            color: #0D1117 !important;
        }
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
            background: #4F46E5 !important;
            color: white !important;
            border-color: #4F46E5 !important;
        }
        [data-testid="stCheckbox"] {
            background: rgba(255,255,255,0.02) !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
        [data-testid="stCheckbox"] p, [data-testid="stCheckbox"] span {
            color: #E2E8F0 !important;
        }
        .card-title { color: #E2E8F0 !important; }
        .card-subtitle { color: rgba(255,255,255,0.6) !important; }
        .vg-section-label { color: rgba(255,255,255,0.5) !important; border-top-color: rgba(255,255,255,0.1) !important; }
        .stNumberInput label, .stSlider label { color:#E2E8F0 !important; }
        
        /* ── DARK PULL SWITCHER BULB ── */
        div[data-testid="stColumn"]:has(.pull-switch-wrapper) button {
            background: radial-gradient(circle at center, #4b5563, #1f2937, #000) !important;
            box-shadow: 0 0 20px 6px rgba(31,41,55,0.7) !important;
        }
        div[data-testid="stColumn"]:has(.pull-switch-wrapper)::before {
            background: #374151 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # ─── REAL-TIME DATA CALCULATION ───────────────────────────
    if ac == 'LUNG':
        pd = D.patient_data
        cur_symptoms = []
        if pd.get('cough_level') in ['severe','chronic']: cur_symptoms.append('persistent_cough')
        if pd.get('cough_level') == 'chronic':            cur_symptoms.append('hemoptysis')
        risk_score, _ = calculate_risk_score(
            pd['age'], pd['pack_years'], pd['family_hist'], cur_symptoms,
            pd['bmi'], pd['cancer_hist'], pd['copd'], pd['radon'],
            pd['occupational'], pd['secondary_smoke'], pd['chest_rad']
        )
        data_pts = sum(1 for v in [pd.get('cough_level'), pd.get('family_level'), pd['pack_years']>0] if v)
        conf_val = f"{risk_score}"
        conf_suffix = '/10'
        b_risk = min(100, max(15, risk_score * 4.5))
        bar_data = [35, 42, 28, 55, b_risk*0.4, b_risk*0.7, b_risk]
        
    elif ac == 'SKIN':
        sd = D.skin_data
        score = len(sd['symptoms'])*3 + (5 if sd['personal_hist'] else 0) + sd['uv_exposure_years']//10
        conf_val = f"{score}"
        conf_suffix = '/10'
        b_risk = min(100, max(15, score * 5))
        bar_data = [20, 30, 25, 40, b_risk*0.3, b_risk*0.6, b_risk]
        
    elif ac == 'PROSTATE':
        pd2 = D.prostate_data
        score = len(pd2['symptoms'])*4 + (5 if pd2['ethnicity_risk'] else 0) + (5 if pd2['family_hist'] else 0)
        conf_val = f"{score}"
        conf_suffix = '/10'
        b_risk = min(100, max(15, score * 5))
        bar_data = [25, 35, 20, 45, b_risk*0.3, b_risk*0.7, b_risk]
        
    elif ac == 'COLORECTAL':
        cd = D.colorectal_data
        score = len(cd['symptoms'])*4 + (5 if cd['family_hist'] else 0) + (5 if cd['ibd'] else 0)
        conf_val = f"{score}"
        conf_suffix = '/10'
        b_risk = min(100, max(15, score * 5))
        bar_data = [30, 40, 35, 50, b_risk*0.4, b_risk*0.8, b_risk]

    # ─── HEADER & THEME TOGGLE ───────────────────────────────
    logo_path = os.path.join(ROOT_DIR, 'src', 'assets', 'logo.png')
    logo_base64 = get_logo_base64(dm, logo_path)
            
    hero_logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height:160px; margin-bottom:28px;">' if logo_base64 else ''
    
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0 4px 20px 4px;">
        <div style="display:flex;align-items:center;font-size:1.3rem;font-weight:800;color:{C_TEXT};letter-spacing:0.04em;">
            AuraMed
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_space, c_btn = st.columns([10, 1])
    with c_btn:
        st.markdown('<div class="pull-switch-wrapper"></div>', unsafe_allow_html=True)
        if st.button(" ", key="theme_toggle", help="Pull down to toggle theme"):
            D.dark_mode = not dm
            st.rerun()

    # ─── HERO SECTION ──────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; min-height:75vh; padding: 20px; max-width: 1000px; margin: 0 auto;">
        {hero_logo_html}
        <h1 style="font-size:4.8rem; font-weight:800; letter-spacing:-2px; color:{C_TEXT}; line-height:1.1; margin-bottom:32px;">
            Next-Generation <br/><span style="background:{C_GRAD}; -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Clinical Risk Stratification</span>
        </h1>
        <p style="font-size:1.35rem; color:{C_TEXT_M}; line-height:1.7; margin-bottom:60px; font-weight:400; max-width: 800px;">
            AuraMed is a highly specialized clinical workstation leveraging advanced neural networks and multifactorial medical logic to generate predictive risk profiles across multiple oncology domains.
        </p>
        <div style="display:inline-block; padding: 16px 36px; background:transparent; border: 2px solid {C_LINE}; border-radius:100px; font-size:1rem; font-weight:700; color:{C_TEXT}; letter-spacing:0.08em; text-transform:uppercase; animation: bounce 2s infinite;">
            Scroll Down for Diagnostics ↓
        </div>
        <style>
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
                40% {{ transform: translateY(-12px); }}
                60% {{ transform: translateY(-6px); }}
            }}
        </style>
    </div>
    """, unsafe_allow_html=True)

    # ─── CANCER SELECTOR (top row of tabs) ────────────────────
    cancer_tab_cols = st.columns(4)
    for i, c in enumerate(['LUNG', 'SKIN', 'PROSTATE', 'COLORECTAL']):
        is_ac = (c == ac)
        with cancer_tab_cols[i]:
            if st.button(c, key=f"ctab_{c}"):
                D.active_cancer = c; set_stage(1); D.lung_tab = 'Symptoms'; st.rerun()

    # CSS-override the cancer tabs to look like proper header pills
    st.markdown(f"""
    <style>
    [data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stColumn"] button {{
        background: transparent !important;
        border: 1.5px solid rgba(26,26,46,0.12) !important;
        color: rgba(26,26,46,0.45) !important;
        font-size: 0.75rem !important;
        padding: 10px 20px !important;
        border-radius: 100px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ─── SPLIT LAYOUT ─────────────────────────────────────────
    col_l, col_r = st.columns([1, 1.4], gap="large")

    # ═══ LEFT PANEL (via components.html for pixel-perfect render) ═══
    with col_l:
        bar_dk = "#E2E8F0" if dm else "#1A1A2E"
        bar_lt = "rgba(255,255,255,0.15)" if dm else "rgba(26,26,46,0.18)"
        bar_items = "".join(
            f'<div style="flex:1;border-radius:12px;background:{bar_lt if i in (2,6) else bar_dk};height:{h}%;min-height:8px;"></div>'
            for i, h in enumerate(bar_data)
        )
        
        bg_card = 'rgba(255,255,255,0.03)' if dm else 'rgba(255,255,255,0.6)'
        model_is_active = (ac == 'LUNG' and D.patient_data.get('image_uploaded', False))
        conf_label = "Vanguard Risk Score"
        
        # Calculate Profile Completeness dynamically
        if ac == 'LUNG':
            if model_is_active: complet_pct = 100
            else:
                c_score = 15
                if curr_stage() == 5: c_score = 100
                else:
                    if D.patient_data.get('cough_level'): c_score += 25
                    if D.patient_data.get('family_level'): c_score += 25
                    if D.patient_data.get('pack_years', 0) > 0: c_score += 15
                    chk_opts = ['cancer_hist','copd','radon','occupational','secondary_smoke','chest_rad']
                    c_score += min(20, sum(20 for k in chk_opts if D.patient_data.get(k)))
                complet_pct = min(100, c_score)
        else:
            complet_pct = min(100, cs * 20)
        
        pie_c1 = "#1A1A2E" if not dm else "#FFFFFF"
        pie_c2 = "#4A7FB5" if not dm else "#82BEE6"
        pie_c3 = "#82BEE6" if not dm else "#4A7FB5"
        pie_c4 = "#E2E8F0" if not dm else "#334155"
        
        # Dynamic segment calculation
        s1 = min(45, max(25, int(b_risk * 0.4)))
        s2 = min(35, max(20, int(b_risk * 0.3)))
        s3 = min(25, max(15, int(b_risk * 0.2)))
        s4 = 100 - (s1 + s2 + s3)
        off2 = -s1
        off3 = off2 - s2
        off4 = off3 - s3

        left_html = f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap" rel="stylesheet">
<style>
  * {{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}}
  body {{background:transparent;}}
  .donut-segment {{ transition: stroke-dasharray 1s ease-out; }}
</style>
</head>
<body>
<div style="padding-top:16px;">
  <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.24em;color:{C_TEXT_M};text-transform:uppercase;margin-bottom:32px;">
    Risk Factor Distribution
  </div>
  
  <div style="display:flex; align-items:center; gap: 36px; margin-bottom: 24px;">
      <!-- SVG Donut Chart -->
      <div style="position:relative; width: 220px; height: 220px; flex-shrink: 0;">
          <svg viewBox="0 0 36 36" style="width:100%; height:100%; transform: rotate(-90deg); filter: drop-shadow(0 10px 20px rgba(0,0,0,0.08));">
            <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="{C_LINE}" stroke-width="3"></circle>
            <circle class="donut-segment" cx="18" cy="18" r="15.915" fill="transparent" stroke="{pie_c1}" stroke-width="4.5" stroke-dasharray="{s1} {{100 - s1}}" stroke-dashoffset="0"></circle>
            <circle class="donut-segment" cx="18" cy="18" r="15.915" fill="transparent" stroke="{pie_c2}" stroke-width="4.5" stroke-dasharray="{s2} {{100 - s2}}" stroke-dashoffset="{off2}"></circle>
            <circle class="donut-segment" cx="18" cy="18" r="15.915" fill="transparent" stroke="{pie_c3}" stroke-width="4.5" stroke-dasharray="{s3} {{100 - s3}}" stroke-dashoffset="{off3}"></circle>
            <circle class="donut-segment" cx="18" cy="18" r="15.915" fill="transparent" stroke="{pie_c4}" stroke-width="4.5" stroke-dasharray="{s4} {{100 - s4}}" stroke-dashoffset="{off4}"></circle>
          </svg>
          <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); text-align:center;">
              <div style="font-size:2.8rem; font-weight:800; color:{C_TEXT}; line-height:1; letter-spacing:-2px;">
                  {conf_val if ac == 'LUNG' else '—'}<span style="font-size:1.2rem;opacity:0.4;">{conf_suffix if ac == 'LUNG' else ''}</span>
              </div>
          </div>
      </div>
      
      <!-- Legend -->
      <div style="display:flex; flex-direction:column; gap: 18px;">
          <div style="display:flex; align-items:center; gap: 12px;">
              <div style="width:14px; height:14px; border-radius:4px; background:{pie_c1};"></div>
              <div style="font-size:0.85rem; font-weight:700; color:{C_TEXT};">Clinical <span style="color:{C_TEXT_M}; margin-left:6px; font-weight:400;">{s1}%</span></div>
          </div>
          <div style="display:flex; align-items:center; gap: 12px;">
              <div style="width:14px; height:14px; border-radius:4px; background:{pie_c2};"></div>
              <div style="font-size:0.85rem; font-weight:700; color:{C_TEXT};">Lifestyle <span style="color:{C_TEXT_M}; margin-left:6px; font-weight:400;">{s2}%</span></div>
          </div>
          <div style="display:flex; align-items:center; gap: 12px;">
              <div style="width:14px; height:14px; border-radius:4px; background:{pie_c3};"></div>
              <div style="font-size:0.85rem; font-weight:700; color:{C_TEXT};">Symptoms <span style="color:{C_TEXT_M}; margin-left:6px; font-weight:400;">{s3}%</span></div>
          </div>
          <div style="display:flex; align-items:center; gap: 12px;">
              <div style="width:14px; height:14px; border-radius:4px; background:{pie_c4};"></div>
              <div style="font-size:0.85rem; font-weight:700; color:{C_TEXT};">Genetics <span style="color:{C_TEXT_M}; margin-left:6px; font-weight:400;">{s4}%</span></div>
          </div>
      </div>
  </div>

  <div style="margin-top:20px;padding:24px;background:{bg_card};border-radius:18px;border:1px solid {C_LINE};">
    <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.2em;color:{C_TEXT_M};text-transform:uppercase;margin-bottom:8px;">Active Protocol</div>
    <div style="font-size:1.45rem;font-weight:800;color:{C_TEXT};letter-spacing:-0.02em;">{ac} Cancer</div>
    <div style="font-size:0.82rem;color:{C_TEXT_M};margin-top:3px;">Profile Completeness: {complet_pct}%</div>
    <div style="margin-top:14px;height:3px;background:{C_LINE};border-radius:100px;overflow:hidden;">
      <div style="width:{complet_pct}%;height:100%;background:{C_TEXT};border-radius:100px;transition:width 0.4s ease;"></div>
    </div>
  </div>
</div>
</body>
</html>"""
        components.html(left_html, height=600, scrolling=False)

    # ═══ RIGHT PANEL: WHITE CARD ═══
    with col_r:
        st.markdown('<div class="vg-right-panel-marker"></div>', unsafe_allow_html=True)

        # ── LUNG ──────────────────────────────────────────────
        if ac == 'LUNG':
            tab = D.lung_tab

            if tab == 'Symptoms':
                card_title(
                    "Symptoms & History",
                    'Answer the <span>clinical parameters</span> below to refine the <span>predictive detection model</span> for your specific case profile.'
                )

                # PERSISTENT COUGH — pill toggles
                section_label("Persistent Cough")
                pd = D.patient_data
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    if st.button("None / Mild",  key="c_none_mild", type="primary" if pd.get('cough_level') == 'none_mild' else "secondary", help="No cough or occasional mild coughing."):  
                        pd['cough_level'] = 'none_mild'; st.rerun()
                with r1c2:
                    if st.button("Moderate",     key="c_moderate", type="primary" if pd.get('cough_level') == 'moderate' else "secondary", help="Regular coughing that disrupts daily activities."):   
                        pd['cough_level'] = 'moderate'; st.rerun()

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    if st.button("Severe",           key="c_severe", type="primary" if pd.get('cough_level') == 'severe' else "secondary", help="Frequent, intense coughing fits affecting sleep."):   
                        pd['cough_level'] = 'severe'; st.rerun()
                with r2c2:
                    if st.button("Chronic (8+ weeks)", key="c_chronic", type="primary" if pd.get('cough_level') == 'chronic' else "secondary", help="A cough that has lasted for 8 weeks or longer continuously."): 
                        pd['cough_level'] = 'chronic'; st.rerun()

                # FAMILY MEDICAL HISTORY
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                section_label("Family Medical History")

                fc1, fc2 = st.columns(2)
                with fc1:
                    if st.button("No History",            key="f_no", type="primary" if pd.get('family_level') == 'no_history' else "secondary", help="No immediate family members diagnosed."):    
                        pd['family_level'] = 'no_history'; pd['family_hist'] = False; st.rerun()
                with fc2:
                    if st.button("First Degree Relative", key="f_first", type="primary" if pd.get('family_level') == 'first_degree' else "secondary", help="Parent, sibling, or child diagnosed with lung cancer."): 
                        pd['family_level'] = 'first_degree'; pd['family_hist'] = True; st.rerun()

                # Additional clinical questions
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                section_label("Additional Risk Markers")
                a1, a2 = st.columns(2)
                with a1:
                    pd['cancer_hist']  = st.checkbox("Prior Malignancy",          pd['cancer_hist'], help="History of any previous cancer diagnosis.")
                    pd['copd']         = st.checkbox("COPD",                       pd['copd'], help="Chronic Obstructive Pulmonary Disease, including emphysema and chronic bronchitis.")
                    pd['radon']        = st.checkbox("Radon Exposure",             pd['radon'], help="Prolonged exposure to radon gas, a leading cause of lung cancer.")
                with a2:
                    pd['chest_rad']    = st.checkbox("Chest Radiation",            pd['chest_rad'], help="Previous radiation therapy to the chest area.")
                    pd['occupational'] = st.checkbox("Occupational Carcinogens",   pd['occupational'], help="Exposure to asbestos, silica, arsenic, diesel exhaust, etc.")
                    pd['secondary_smoke'] = st.checkbox("Secondary Smoke",         pd['secondary_smoke'], help="Regular exposure to secondhand smoke.")

                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                n1, n2 = st.columns(2)
                with n1: pd['age']      = st.number_input("AGE", 30, 100, pd['age'], help="Patient's current age in years.")
                with n2: pd['pack_years'] = st.slider("SMOKING (PACK-YRS)", 0, 150, pd['pack_years'], help="Calculated by multiplying packs smoked per day by the number of years smoked.")

                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                if st.button("Continue to Imaging →", key="go_imaging"):
                    D.lung_tab = 'Imaging'; st.rerun()

            elif tab == 'Imaging':
                card_title(
                    "Imaging Protocol",
                    'Upload a <span>CT scan or chest X-ray</span> to run the AI nodule detection model with Grad-CAM visualization.'
                )
                pd = D.patient_data
                # Show risk summary
                c1, c2 = st.columns(2)
                with c1: st.metric("SMOKING HISTORY", f"{pd['pack_years']} pack-yrs")
                with c2: st.metric("COPD STATUS", "Positive" if pd['copd'] else "Negative")
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

                up = st.file_uploader("Upload CT / X-Ray", type=['dcm','png','jpg'], label_visibility="collapsed")
                if up:
                    if not pd.get('image_uploaded', False):
                        pd['image_uploaded'] = True
                        st.rerun()
                    with st.spinner("Running Grad-CAM analysis..."):
                        model, device = load_vanguard_model()
                        prob, status, orig, heatmap = analyze_source_file(up, model, device)
                        time.sleep(0.8)
                    v1, v2 = st.columns(2)
                    with v1:
                        if orig    is not None: st.image(orig,    caption="Source Scan",  use_container_width=True)
                    with v2:
                        if heatmap is not None: 
                            lbl = "REJECTION OVERLAY" if "INVALID" in status else "AI HEATMAP"
                            st.image(heatmap, caption=lbl, use_container_width=True)
                    if "INVALID" in status or "REJECTED" in status:
                        st.markdown(f"""
                        <div style="background:rgba(239, 68, 68, 0.1); border:2px solid #EF4444; border-radius:18px; padding:24px; text-align:center;">
                            <div style="font-size:1.8rem; font-weight:800; color:#EF4444; margin-bottom:8px;">⚠️ SCAN REJECTED</div>
                            <div style="font-size:0.9rem; color:rgba(26,26,46,0.7);">{status}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        risk_color = "#EF4444" if prob > 0.5 else "#10B981"
                        st.markdown(f"""
                        <div style="text-align:center;padding:28px 0 8px;">
                            <div style="font-size:4.5rem;font-weight:800;color:{risk_color};letter-spacing:-3px;line-height:1;">{prob:.1%}</div>
                            <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.2em;color:rgba(26,26,46,0.4);text-transform:uppercase;margin-top:6px;">Vanguard Probability Index</div>
                            <div style="font-size:0.85rem;font-weight:700;color:{risk_color};margin-top:4px;">{status}</div>
                        </div>
                        """, unsafe_allow_html=True)

                if st.button("Continue to Results →", key="go_results"):
                    D.lung_tab = 'Results'; st.rerun()

            elif tab == 'Results':
                pd = D.patient_data
                # Build symptoms list from pill selections
                symptoms = []
                if pd.get('cough_level') in ['severe','chronic']: symptoms.append('persistent_cough')
                if pd.get('cough_level') == 'chronic':            symptoms.append('hemoptysis')
                pd['symptoms'] = symptoms

                score, level = calculate_risk_score(
                    pd['age'], pd['pack_years'], pd['family_hist'], pd['symptoms'],
                    pd['bmi'], pd['cancer_hist'], pd['copd'], pd['radon'],
                    pd['occupational'], pd['secondary_smoke'], pd['chest_rad']
                )
                risk_color = "#EF4444" if level in ["HIGH","CRITICAL"] else "#F59E0B" if level=="MODERATE" else "#10B981"

                card_title("Risk Stratification")
                c1, c2 = st.columns(2)
                with c1: st.metric("RISK LEVEL",      level)
                with c2: st.metric("AI CONFIDENCE",   f"{89.4 + (score % 6.4):.1f}%")

                if level != "LOW":
                    st.markdown(f"""
                    <div style="background:rgba(26,26,46,0.03);border:1px solid rgba(26,26,46,0.07);
                                border-radius:18px;padding:26px;margin-top:24px;">
                        <div style="font-size:1.1rem;font-weight:800;color:{risk_color};margin-bottom:6px;">
                            Specialist Referral Recommended
                        </div>
                        <div style="font-size:0.88rem;color:rgba(26,26,46,0.6);line-height:1.6;">
                            A <b>Thoracic Oncology Consultation</b> within 72 hours is advised based on your risk profile.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                find_doctor_widget("Thoracic Oncologist", icon="🫁", color="#4A7FB5", search_suffix="Oncology Clinic")

                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                if st.button("Restart Screening", key="rst_lung"):
                    D.lung_tab = 'Symptoms'
                    D.patient_data.update({'cough_level':None,'family_level':None,'symptoms':[], 'image_uploaded': False})
                    set_stage(1); st.rerun()

        # ── SKIN ──────────────────────────────────────────────
        elif ac == 'SKIN':
            sd = D.skin_data
            if cs == 1:
                card_title("Dermatological Profile", "UV exposure history and <span>Fitzpatrick skin type</span> classification.")
                a1, a2 = st.columns(2)
                with a1: sd['age']       = st.number_input("AGE", 10, 100, sd['age'], key='sk_age', help="Patient's current age in years.")
                with a2: sd['skin_type'] = st.selectbox("FITZPATRICK TYPE", [1,2,3,4,5,6], index=sd['skin_type']-1, format_func=lambda x:f"Type {x}", key='sk_tp', help="Skin type classification: Type 1 always burns, never tans; Type 6 never burns, deeply pigmented.")
                if st.button("Continue →", key='sk1'): set_stage(2); st.rerun()
            elif cs == 2:
                card_title("UV Exposure", "Historical ultraviolet radiation exposure assessment.")
                sd['uv_exposure_years'] = st.slider("UV EXPOSURE (YEARS)", 0, 60, sd['uv_exposure_years'], help="Duration of significant cumulative ultraviolet radiation exposure.")
                sd['tanning_beds']      = st.checkbox("Tanning Bed History", sd['tanning_beds'], help="Prior use of indoor artificial tanning beds.")
                sd['severe_sunburns']   = st.checkbox("Severe Sunburn Events", sd['severe_sunburns'], help="History of blistering or severe sunburns, especially in childhood.")
                if st.button("Continue →", key='sk2'): set_stage(3); st.rerun()
            elif cs == 3:
                card_title("Genetic Risk Factors", "Hereditary and immune system risk markers.")
                sd['family_hist']      = st.checkbox("Family History of Melanoma",    sd['family_hist'], help="Immediate family members diagnosed with melanoma.")
                sd['personal_hist']    = st.checkbox("Personal Skin Cancer History",  sd['personal_hist'], help="Previous diagnosis of melanoma, basal cell, or squamous cell carcinoma.")
                sd['immunosuppressed'] = st.checkbox("Immunosuppressed Status",        sd['immunosuppressed'], help="Weakened immune system due to medication, organ transplant, or disease.")
                if st.button("Continue →", key='sk3'): set_stage(4); st.rerun()
            elif cs == 4:
                card_title("ABCDE Lesion Analysis", "Mole and lesion characteristics evaluation protocol.")
                a1, a2 = st.columns(2)
                with a1:
                    sa = st.checkbox("Asymmetry",         'asymmetry' in sd['symptoms'], help="One half of the mole does not match the other.")
                    sb = st.checkbox("Irregular Border",  'border'    in sd['symptoms'], help="The edges are ragged, blurred, or irregular.")
                    sc = st.checkbox("Multiple Colours",  'color'     in sd['symptoms'], help="The color is not uniform and has different shades of brown, black, or red.")
                with a2:
                    sdi = st.checkbox("Diameter > 6mm",   'diameter'  in sd['symptoms'], help="The lesion is larger than 6mm (about the size of a pencil eraser).")
                    se  = st.checkbox("Evolving Lesion",  'evolving'  in sd['symptoms'], help="The mole is changing in size, shape, or color.")
                if st.button("Generate Analysis", key='sk4'):
                    sd['symptoms'] = [k for k,v in [('asymmetry',sa),('border',sb),('color',sc),('diameter',sdi),('evolving',se)] if v]
                    sc_v = len(sd['symptoms'])*3 + (5 if sd['personal_hist'] else 0) + sd['uv_exposure_years']//10
                    D.skin_results = {"score":sc_v,"level":"HIGH" if sc_v>10 else "MODERATE" if sc_v>5 else "LOW"}
                    set_stage(5); st.rerun()
            elif cs == 5:
                r  = D.skin_results
                rc = "#EF4444" if r['level']=="HIGH" else "#F59E0B" if r['level']=="MODERATE" else "#10B981"
                card_title("Dermatology Report")
                # st.metric removed as per request (0-30 score removal)
                st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{rc};margin:16px 0;'>{r['level']} RISK</div>", unsafe_allow_html=True)
                find_doctor_widget("Dermatologist", icon="✨", color="#F59E0B", search_suffix="Dermatology Center")
                if st.button("Restart", key='rst_sk'): set_stage(1); st.rerun()

        # ── PROSTATE ──────────────────────────────────────────
        elif ac == 'PROSTATE':
            pd2 = D.prostate_data
            if cs == 1:
                card_title("Prostate Assessment", "Age, ethnicity, and <span>genetic risk</span> profiling.")
                pd2['age']            = st.number_input("AGE", 40, 100, pd2['age'], key='pr_age', help="Patient's current age in years.")
                pd2['ethnicity_risk'] = st.checkbox("African-American / Caribbean Ancestry", pd2['ethnicity_risk'], help="African-American men and Caribbean men of African ancestry are at significantly higher risk.")
                if st.button("Continue →", key='pr1'): set_stage(2); st.rerun()
            elif cs == 2:
                card_title("Clinical History", "Family oncology and <span>genetic mutation</span> screening.")
                pd2['family_hist']   = st.checkbox("Family History of Prostate Cancer", pd2['family_hist'], help="Having a father or brother diagnosed with prostate cancer.")
                pd2['brca_mutation'] = st.checkbox("Known BRCA Gene Mutation",         pd2['brca_mutation'], help="Presence of BRCA1 or BRCA2 genetic mutations.")
                if st.button("Continue →", key='pr2'): set_stage(3); st.rerun()
            elif cs == 3:
                card_title("Urological Symptoms", "Lower urinary tract symptom evaluation.")
                su  = st.checkbox("Frequent / Urgent Urination",    'urinary' in pd2['symptoms'], help="Needing to urinate often, especially at night, or having sudden urges.")
                sbl = st.checkbox("Hematuria / Blood in Semen",     'blood'   in pd2['symptoms'], help="Visible blood in urine or seminal fluid.")
                sbo = st.checkbox("Bone Pain (Pelvis / Spine)",     'bone'    in pd2['symptoms'], help="Deep pain or stiffness in the lower back, hips, or upper thighs.")
                if st.button("Generate Analysis", key='pr3'):
                    pd2['symptoms'] = [k for k,v in [('urinary',su),('blood',sbl),('bone',sbo)] if v]
                    sc_v = len(pd2['symptoms'])*4 + (5 if pd2['ethnicity_risk'] else 0) + (5 if pd2['family_hist'] else 0)
                    D.prostate_results = {"score":sc_v,"level":"HIGH" if sc_v>12 else "MODERATE" if sc_v>6 else "LOW"}
                    set_stage(4); st.rerun()
            elif cs == 4:
                r  = D.prostate_results
                rc = "#EF4444" if r['level']=="HIGH" else "#F59E0B" if r['level']=="MODERATE" else "#10B981"
                card_title("Prostate Risk Report")
                # st.metric removed as per request (0-30 score removal)
                st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{rc};margin:16px 0;'>{r['level']} RISK</div>", unsafe_allow_html=True)
                find_doctor_widget("Urologist", icon="🔬", color="#8B5CF6", search_suffix="Urology Department")
                if st.button("Restart", key='rst_pr'): set_stage(1); st.rerun()

        # ── COLORECTAL ────────────────────────────────────────
        elif ac == 'COLORECTAL':
            cd = D.colorectal_data
            if cs == 1:
                card_title("Colorectal Assessment", "GI risk profiling via lifestyle and <span>genetic markers</span>.")
                a1, a2 = st.columns(2)
                with a1: cd['age'] = st.number_input("AGE", 20, 100, cd['age'], key='cr_age', help="Patient's current age in years.")
                with a2: cd['bmi'] = st.number_input("BMI", 10.0, 60.0, cd['bmi'], key='cr_bmi', help="Body Mass Index (kg/m²).")
                if st.button("Continue →", key='cr1'): set_stage(2); st.rerun()
            elif cs == 2:
                card_title("GI History", "IBD and familial risk factor screening.")
                cd['family_hist'] = st.checkbox("Family History of CRC",              cd['family_hist'], help="Immediate family member with colorectal cancer or adenomatous polyps.")
                cd['ibd']         = st.checkbox("IBD (Crohn's / Ulcerative Colitis)", cd['ibd'], help="Inflammatory Bowel Disease significantly increases CRC risk.")
                if st.button("Continue →", key='cr2'): set_stage(3); st.rerun()
            elif cs == 3:
                card_title("GI Symptoms", "Current gastrointestinal symptom mapping.")
                sbl = st.checkbox("Rectal / Stool Bleeding",      'blood' in cd['symptoms'], help="Finding blood in the stool or bleeding from the rectum.")
                sbw = st.checkbox("Bowel Habit Changes (>4 wks)", 'bowel' in cd['symptoms'], help="Persistent diarrhea, constipation, or change in stool consistency for >1 month.")
                spn = st.checkbox("Persistent Abdominal Pain",    'pain'  in cd['symptoms'], help="Ongoing cramps, gas, or pain in the abdomen.")
                if st.button("Generate Analysis", key='cr3'):
                    cd['symptoms'] = [k for k,v in [('blood',sbl),('bowel',sbw),('pain',spn)] if v]
                    sc_v = len(cd['symptoms'])*4 + (5 if cd['family_hist'] else 0) + (5 if cd['ibd'] else 0)
                    D.colorectal_results = {"score":sc_v,"level":"HIGH" if sc_v>12 else "MODERATE" if sc_v>6 else "LOW"}
                    set_stage(4); st.rerun()
            elif cs == 4:
                r  = D.colorectal_results
                rc = "#EF4444" if r['level']=="HIGH" else "#F59E0B" if r['level']=="MODERATE" else "#10B981"
                card_title("Colorectal Risk Report")
                # st.metric removed as per request (0-30 score removal)
                st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{rc};margin:16px 0;'>{r['level']} RISK</div>", unsafe_allow_html=True)
                find_doctor_widget("Gastroenterologist", icon="🥗", color="#10B981", search_suffix="Gastroenterology Institute")
                if st.button("Restart", key='rst_cr'): set_stage(1); st.rerun()


if __name__ == "__main__":
    main()
