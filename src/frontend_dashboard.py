"""
AuraScan Frontend Dashboard — SHOWCASE / DEMO MODE
====================================================
This is the showcase version of the dashboard. The AI model and risk scoring
calls have been replaced with static demo data so the UI can be explored
without requiring model weights or patient data.

The full system integrates:
  - NoduleClassifier3D  (src/ai_model_architecture.py)
  - calculate_risk_score (src/clinical_risk_calculator.py)
  - DICOM preprocessing  (src/preprocessing.py)
  - Grad-CAM heatmaps    (via PyTorch hooks on conv3)
"""

import streamlit as st
import streamlit.components.v1 as components
import time

st.set_page_config(
    page_title="AuraScan — Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── DEMO NOTICE ────────────────────────────────────────────────
st.markdown("""
<div style="background:#FFF3CD;border:1px solid #FBBF24;border-radius:12px;
            padding:14px 20px;margin-bottom:24px;font-size:0.88rem;font-weight:600;">
    🔒 <b>Showcase Mode:</b> AI inference and risk scoring are disabled.
    This demo shows the UI architecture and workflow only.
    Model weights and proprietary logic are not included in this repository.
</div>
""", unsafe_allow_html=True)

# ── PREMIUM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #EEF1F6 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 28px 52px 140px 52px !important; max-width: 1480px !important; }
.card-title { font-size: 2.1rem; font-weight: 800; color: #1A1A2E; letter-spacing:-0.03em; margin-bottom:6px; }
.card-subtitle { font-size: 0.9rem; color: rgba(26,26,46,0.5); line-height:1.65; margin-bottom:30px; }
.vg-section-label { font-size:0.65rem; font-weight:700; letter-spacing:0.22em; text-transform:uppercase;
                    color:rgba(26,26,46,0.38); margin-bottom:14px; }
div.stButton > button { background:transparent !important; color:#1A1A2E !important;
    border-radius:100px !important; border:1.5px solid rgba(26,26,46,0.18) !important;
    padding:14px 28px !important; font-weight:600 !important; width:100% !important;
    transition:all 0.25s ease !important; }
div.stButton > button:hover { background:#1A1A2E !important; color:white !important; }
[data-testid="stMetric"] { background:rgba(26,26,46,0.03) !important;
    padding:20px !important; border-radius:16px !important; border:1px solid rgba(26,26,46,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────
D = st.session_state
if 'active_cancer'    not in D: D.active_cancer    = 'LUNG'
if 'lung_tab'         not in D: D.lung_tab          = 'Symptoms'
if 'intake_stage'     not in D: D.intake_stage      = 1
if 'skin_stage'       not in D: D.skin_stage         = 1
if 'prostate_stage'   not in D: D.prostate_stage     = 1
if 'colorectal_stage' not in D: D.colorectal_stage   = 1
if 'patient_data' not in D:
    D.patient_data = {
        'age': 55, 'bmi': 24.5, 'pack_years': 0,
        'family_hist': False, 'cancer_hist': False, 'copd': False,
        'chest_rad': False, 'radon': False, 'occupational': False,
        'secondary_smoke': False, 'cough_level': None, 'family_level': None,
    }

STAGE_KEYS = {
    'LUNG': 'intake_stage', 'SKIN': 'skin_stage',
    'PROSTATE': 'prostate_stage', 'COLORECTAL': 'colorectal_stage'
}
def curr_stage(): return getattr(D, STAGE_KEYS[D.active_cancer])
def set_stage(v):  setattr(D, STAGE_KEYS[D.active_cancer], v)

# ── DEMO RISK SCORE (replaces the real calculate_risk_score) ───
def demo_risk_score(patient_data):
    """
    Static demo scorer. In the full system this calls:
        from src.clinical_risk_calculator import calculate_risk_score
    which applies the PLCOm2012-inspired weighted factor model.
    """
    pd = patient_data
    score = 0
    if pd.get('age', 0) > 45:           score += 1
    if pd.get('pack_years', 0) > 0:     score += 1
    if pd.get('family_hist'):           score += 1
    if pd.get('cough_level') in ['severe', 'chronic']: score += 1
    if pd.get('cancer_hist'):           score += 1
    if pd.get('copd'):                  score += 1
    if pd.get('chest_rad'):             score += 1
    if pd.get('radon'):                 score += 1
    if pd.get('occupational'):          score += 1
    if pd.get('secondary_smoke'):       score += 1
    if score >= 8:   level = "CRITICAL"
    elif score >= 6: level = "HIGH"
    elif score >= 3: level = "MODERATE"
    else:            level = "LOW"
    return score, level

# ── HELPERS ────────────────────────────────────────────────────
def section_label(txt):
    st.markdown(f'<div class="vg-section-label">{txt}</div>', unsafe_allow_html=True)

def card_title(title, subtitle=None):
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def find_doctor_widget(specialty, icon="📍", color="#4A7FB5"):
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    section_label(f"Locate {specialty} Professionals")
    loc = st.text_input("Enter your City or ZIP", placeholder="e.g., New York, NY",
                        key=f"loc_{specialty.replace(' ','_')}")
    if loc:
        query = f"{specialty} near {loc}".replace(" ", "+")
        st.markdown(f"""
        <div style="background:white;border-radius:20px;border:1px solid rgba(26,26,46,0.08);
                    padding:32px;margin-top:12px;text-align:center;">
            <div style="font-size:3rem;margin-bottom:16px;">{icon}</div>
            <div style="font-size:1.2rem;font-weight:800;color:#1A1A2E;margin-bottom:8px;">
                Find {specialty}s in {loc}</div>
            <a href="https://www.google.com/maps/search/{query}" target="_blank"
               style="text-decoration:none;">
                <div style="background:{color};color:white;padding:14px 32px;border-radius:100px;
                            font-weight:700;display:inline-block;margin-top:12px;">
                    View on Google Maps</div></a>
        </div>""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────
ac  = D.active_cancer
cs  = curr_stage()
pd_data = D.patient_data

st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0 4px 20px 4px;">
    <div style="font-size:1.3rem;font-weight:800;color:#1A1A2E;letter-spacing:0.04em;">
        AuraScan
    </div>
    <div style="font-size:0.75rem;color:rgba(26,26,46,0.4);font-weight:600;
                background:rgba(26,26,46,0.05);padding:6px 14px;border-radius:100px;">
        SHOWCASE DEMO
    </div>
</div>
""", unsafe_allow_html=True)

# ── CANCER MODULE TABS ─────────────────────────────────────────
tab_cols = st.columns(4)
for i, c in enumerate(['LUNG', 'SKIN', 'PROSTATE', 'COLORECTAL']):
    with tab_cols[i]:
        if st.button(c, key=f"ctab_{c}",
                     type="primary" if c == ac else "secondary"):
            D.active_cancer = c; set_stage(1); st.rerun()

st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# ── SPLIT LAYOUT ───────────────────────────────────────────────
score, level = demo_risk_score(pd_data)
risk_color   = "#EF4444" if level in ["HIGH","CRITICAL"] else "#F59E0B" if level=="MODERATE" else "#10B981"
b_risk       = min(100, max(15, score * 10))

col_l, col_r = st.columns([1, 1.4], gap="large")

# ── LEFT: Analytics Panel ──────────────────────────────────────
with col_l:
    s1 = min(45, max(25, int(b_risk * 0.40)))
    s2 = min(35, max(20, int(b_risk * 0.30)))
    s3 = min(25, max(15, int(b_risk * 0.20)))
    s4 = 100 - (s1 + s2 + s3)
    off2, off3 = -s1, -s1 - s2

    components.html(f"""
    <!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap" rel="stylesheet">
    <style>* {{margin:0;padding:0;box-sizing:border-box;font-family:'Inter',sans-serif;}}</style>
    </head><body>
    <div style="padding-top:16px;">
      <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.24em;
                  color:rgba(26,26,46,0.5);text-transform:uppercase;margin-bottom:32px;">
        Risk Factor Distribution</div>
      <div style="display:flex;align-items:center;gap:36px;margin-bottom:24px;">
        <div style="position:relative;width:220px;height:220px;flex-shrink:0;">
          <svg viewBox="0 0 36 36" style="width:100%;height:100%;transform:rotate(-90deg);
               filter:drop-shadow(0 10px 20px rgba(0,0,0,0.08));">
            <circle cx="18" cy="18" r="15.915" fill="transparent"
                    stroke="rgba(26,26,46,0.08)" stroke-width="3"></circle>
            <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#1A1A2E"
                    stroke-width="4.5" stroke-dasharray="{s1} {100-s1}" stroke-dashoffset="0"></circle>
            <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#4A7FB5"
                    stroke-width="4.5" stroke-dasharray="{s2} {100-s2}" stroke-dashoffset="{off2}"></circle>
            <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#82BEE6"
                    stroke-width="4.5" stroke-dasharray="{s3} {100-s3}" stroke-dashoffset="{off3}"></circle>
            <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#E2E8F0"
                    stroke-width="4.5" stroke-dasharray="{s4} {100-s4}" stroke-dashoffset="{off3-s3}"></circle>
          </svg>
          <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;">
            <div style="font-size:2.8rem;font-weight:800;color:#1A1A2E;line-height:1;letter-spacing:-2px;">
              {score}<span style="font-size:1.2rem;opacity:0.4;">/10</span></div>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:18px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:14px;height:14px;border-radius:4px;background:#1A1A2E;"></div>
            <div style="font-size:0.85rem;font-weight:700;color:#1A1A2E;">
              Clinical <span style="color:rgba(26,26,46,0.5);margin-left:6px;font-weight:400;">{s1}%</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:14px;height:14px;border-radius:4px;background:#4A7FB5;"></div>
            <div style="font-size:0.85rem;font-weight:700;color:#1A1A2E;">
              Lifestyle <span style="color:rgba(26,26,46,0.5);margin-left:6px;font-weight:400;">{s2}%</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:14px;height:14px;border-radius:4px;background:#82BEE6;"></div>
            <div style="font-size:0.85rem;font-weight:700;color:#1A1A2E;">
              Symptoms <span style="color:rgba(26,26,46,0.5);margin-left:6px;font-weight:400;">{s3}%</span></div>
          </div>
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:14px;height:14px;border-radius:4px;background:#E2E8F0;"></div>
            <div style="font-size:0.85rem;font-weight:700;color:#1A1A2E;">
              Genetics <span style="color:rgba(26,26,46,0.5);margin-left:6px;font-weight:400;">{s4}%</span></div>
          </div>
        </div>
      </div>
      <div style="margin-top:20px;padding:24px;background:rgba(255,255,255,0.6);
                  border-radius:18px;border:1px solid rgba(26,26,46,0.08);">
        <div style="font-size:0.6rem;font-weight:700;letter-spacing:0.2em;
                    color:rgba(26,26,46,0.5);text-transform:uppercase;margin-bottom:8px;">
          Active Protocol</div>
        <div style="font-size:1.45rem;font-weight:800;color:#1A1A2E;">{ac} Cancer</div>
        <div style="font-size:0.82rem;color:rgba(26,26,46,0.5);margin-top:4px;">
          Risk Level: <b style="color:{risk_color};">{level}</b></div>
      </div>
    </div>
    </body></html>""", height=520, scrolling=False)

# ── RIGHT: White Card ──────────────────────────────────────────
with col_r:
    st.markdown("""
    <style>
    div[data-testid="stColumn"]:last-child {
        background:#FFFFFF !important; border-radius:28px !important;
        padding:48px 44px !important;
        box-shadow:0 20px 60px rgba(0,0,0,0.06) !important;
        border:1px solid rgba(0,0,0,0.04) !important;
    }
    </style>""", unsafe_allow_html=True)

    # ── LUNG MODULE ───────────────────────────────────────────
    if ac == 'LUNG':
        tab = D.lung_tab
        tabs = st.columns(3)
        for i, t in enumerate(['Symptoms', 'Imaging', 'Results']):
            with tabs[i]:
                if st.button(t, key=f"lt_{t}", type="primary" if t==tab else "secondary"):
                    D.lung_tab = t; st.rerun()
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        if tab == 'Symptoms':
            card_title("Symptoms & History",
                       'Answer the clinical parameters below to refine the predictive risk model.')
            section_label("Persistent Cough")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("None / Mild", key="c_nm",
                             type="primary" if pd_data.get('cough_level')=='none_mild' else "secondary"):
                    pd_data['cough_level'] = 'none_mild'; st.rerun()
            with c2:
                if st.button("Moderate", key="c_mod",
                             type="primary" if pd_data.get('cough_level')=='moderate' else "secondary"):
                    pd_data['cough_level'] = 'moderate'; st.rerun()
            c3, c4 = st.columns(2)
            with c3:
                if st.button("Severe", key="c_sev",
                             type="primary" if pd_data.get('cough_level')=='severe' else "secondary"):
                    pd_data['cough_level'] = 'severe'; st.rerun()
            with c4:
                if st.button("Chronic (8+ wks)", key="c_chr",
                             type="primary" if pd_data.get('cough_level')=='chronic' else "secondary"):
                    pd_data['cough_level'] = 'chronic'; st.rerun()

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            section_label("Family Medical History")
            f1, f2 = st.columns(2)
            with f1:
                if st.button("No History", key="f_no",
                             type="primary" if pd_data.get('family_level')=='no_history' else "secondary"):
                    pd_data['family_level'] = 'no_history'; pd_data['family_hist'] = False; st.rerun()
            with f2:
                if st.button("First Degree Relative", key="f_1st",
                             type="primary" if pd_data.get('family_level')=='first_degree' else "secondary"):
                    pd_data['family_level'] = 'first_degree'; pd_data['family_hist'] = True; st.rerun()

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            section_label("Additional Risk Markers")
            a1, a2 = st.columns(2)
            with a1:
                pd_data['cancer_hist']    = st.checkbox("Prior Malignancy",         pd_data['cancer_hist'])
                pd_data['copd']           = st.checkbox("COPD",                      pd_data['copd'])
                pd_data['radon']          = st.checkbox("Radon Exposure",            pd_data['radon'])
            with a2:
                pd_data['chest_rad']      = st.checkbox("Chest Radiation",           pd_data['chest_rad'])
                pd_data['occupational']   = st.checkbox("Occupational Carcinogens",  pd_data['occupational'])
                pd_data['secondary_smoke']= st.checkbox("Secondary Smoke",           pd_data['secondary_smoke'])

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            n1, n2 = st.columns(2)
            with n1: pd_data['age']        = st.number_input("AGE", 30, 100, pd_data['age'])
            with n2: pd_data['pack_years'] = st.slider("SMOKING (PACK-YRS)", 0, 150, pd_data['pack_years'])

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            if st.button("Continue to Imaging →", key="go_img"):
                D.lung_tab = 'Imaging'; st.rerun()

        elif tab == 'Imaging':
            card_title("Imaging Protocol",
                       "Upload a CT scan to run the AI nodule detection model with Grad-CAM visualization.")
            c1, c2 = st.columns(2)
            with c1: st.metric("SMOKING HISTORY", f"{pd_data['pack_years']} pack-yrs")
            with c2: st.metric("COPD STATUS", "Positive" if pd_data['copd'] else "Negative")
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            up = st.file_uploader("Upload CT / X-Ray", type=['dcm','png','jpg'],
                                  label_visibility="collapsed")
            if up:
                with st.spinner("Running Grad-CAM analysis..."):
                    time.sleep(1.2)
                st.markdown("""
                <div style="background:rgba(245,158,11,0.08);border:1px solid #F59E0B;
                            border-radius:18px;padding:28px;text-align:center;margin-top:16px;">
                    <div style="font-size:1.8rem;font-weight:800;color:#F59E0B;margin-bottom:8px;">
                        ⚠️ Demo Mode</div>
                    <div style="font-size:0.9rem;color:rgba(26,26,46,0.6);">
                        In the full system, this would run the <b>NoduleClassifier3D</b>
                        model and overlay a <b>Grad-CAM heatmap</b> on your scan.
                        Model weights are not included in this showcase.
                    </div>
                </div>""", unsafe_allow_html=True)
            if st.button("Continue to Results →", key="go_res"):
                D.lung_tab = 'Results'; st.rerun()

        elif tab == 'Results':
            sc, lv = demo_risk_score(pd_data)
            rc = "#EF4444" if lv in ["HIGH","CRITICAL"] else "#F59E0B" if lv=="MODERATE" else "#10B981"
            card_title("Risk Stratification")
            c1, c2 = st.columns(2)
            with c1: st.metric("RISK LEVEL", lv)
            with c2: st.metric("RISK SCORE", f"{sc} / 10")
            if lv != "LOW":
                st.markdown(f"""
                <div style="background:rgba(26,26,46,0.03);border:1px solid rgba(26,26,46,0.07);
                            border-radius:18px;padding:26px;margin-top:24px;">
                    <div style="font-size:1.1rem;font-weight:800;color:{rc};margin-bottom:6px;">
                        Specialist Referral Recommended</div>
                    <div style="font-size:0.88rem;color:rgba(26,26,46,0.6);line-height:1.6;">
                        A <b>Thoracic Oncology Consultation</b> within 72 hours is advised.
                    </div></div>""", unsafe_allow_html=True)
            find_doctor_widget("Thoracic Oncologist", icon="🫁", color="#4A7FB5")
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            if st.button("Restart Screening", key="rst"):
                D.lung_tab = 'Symptoms'
                pd_data.update({'cough_level':None,'family_level':None,'family_hist':False})
                set_stage(1); st.rerun()

    # ── OTHER CANCER MODULES ──────────────────────────────────
    elif ac == 'SKIN':
        card_title("Dermatological Profile",
                   "UV exposure history and Fitzpatrick skin type classification.")
        st.info("🔒 Skin cancer module UI — logic omitted in showcase version.")
        find_doctor_widget("Dermatologist", icon="✨", color="#F59E0B")

    elif ac == 'PROSTATE':
        card_title("Prostate Assessment",
                   "Age, ethnicity, and genetic risk profiling.")
        st.info("🔒 Prostate cancer module UI — logic omitted in showcase version.")
        find_doctor_widget("Urologist", icon="🔬", color="#8B5CF6")

    elif ac == 'COLORECTAL':
        card_title("Colorectal Assessment",
                   "GI risk profiling via lifestyle and genetic markers.")
        st.info("🔒 Colorectal cancer module UI — logic omitted in showcase version.")
        find_doctor_widget("Gastroenterologist", icon="🥗", color="#10B981")


def main():
    pass  # Entry point used when launched via `streamlit run src/frontend_dashboard.py`

if __name__ == "__main__":
    main()
