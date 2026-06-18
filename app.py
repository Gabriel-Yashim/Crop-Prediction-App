import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Crop Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load artefacts
@st.cache_resource
def load_artifacts():
    base = "artifacts"
    model   = joblib.load(os.path.join(base, "crop_model_Random Forest.pkl"))
    scaler  = joblib.load(os.path.join(base, "scaler.pkl"))
    le      = joblib.load(os.path.join(base, "label_encoder.pkl"))
    features = joblib.load(os.path.join(base, "features.pkl"))
    return model, scaler, le, features

try:
    model, scaler, le, FEATURES = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_error = str(e)

# Crop emoji map 
CROP_EMOJI = {
    "rice": "🌾",     "maize": "🌽",        "banana": "🍌",
    "mango": "🥭",    "grapes": "🍇",        "watermelon": "🍉",
    "orange": "🍊",   "muskmelon": "🍈",     "apple": "🍎",
    "coconut": "🥥",  "papaya": ":papaya",        "pawpaw": ":papaya",
    "cotton": "🌿",   "coffee": "☕",        "jute/ewedu": "🌱",
    "potato": "🥔",   "kidneybeans": "🫘",   "blackgram": "🫘",
    "lentil": "🫘",   "mothbeans": "🫘",     "mungbean": "🫘",
    "pigeonpeas": "🫘","chickpea": "🫘",      "cashew": "🌰",
    "palm tree": "🌴","beni/sesame seed": "🌻","black beans": "🫘",
    "jute": "🌱",     "coffee": "☕",
}


def crop_icon(name: str) -> str:
    return CROP_EMOJI.get(name.lower(), "🌿")




# Main area 
st.title("🌱 Smart Crop Recommendation System")
st.markdown("#### By ALFA SIMON ENEKELE  |  Mat. No. 22L1CS0235")
st.markdown(
    "This app uses a machine learning model trained on soil and climate data "
    "to recommend the most suitable crop for your farm conditions."
)
st.markdown("Enter soil and climate conditions to get a crop recommendation.")
st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Soil Nutrients")
    N = st.number_input("Nitrogen (N)", 0, 200, 60,
                    help="Ratio of Nitrogen content in soil (mg/kg)")
    P = st.number_input("Phosphorus (P)", 0, 150, 50,
                    help="Ratio of Phosphorus content in soil (mg/kg)")
    K = st.number_input("Potassium (K)", 0, 210, 40,
                    help="Ratio of Potassium content in soil (mg/kg)")

with col2:
    st.subheader("Climate Conditions")
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
    ph = st.number_input("Soil pH", 3.0, 14.0, 6.5, step=0.01)
    rainfall = st.number_input("Rainfall (mm)", 20.0, 300.0, 100.0, step=0.5)

st.divider()
predict_btn = st.button("Predict Crop", type="primary", use_container_width=True)
if not artifacts_ok:
    st.error(
        f"Could not load model artefacts. "
        f"Please run the notebook first to generate the `artifacts/` folder.\n\n`{artifact_error}`"
    )
    st.stop()


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/wheat.png", width=80)
    # st.title("🌾 Crop Predictor")
    #  Input summary table 
    st.subheader("Your Input Parameters")
    input_data = {
        "Feature":      ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                        "Temperature", "Soil pH", "Rainfall"],
        "Value":        [N, P, K, f"{temperature} °C", ph, f"{rainfall} mm"],
        "Typical Range": ["0–179", "5–145", "5–205",
                        "8–44 °C", "3.5–9.9", "20–298 mm"],
    }
    st.dataframe(pd.DataFrame(input_data), use_container_width=True, hide_index=True)


#  Prediction 
if predict_btn:
    raw = np.array([[N, P, K, temperature, ph, rainfall]])
    scaled = scaler.transform(raw)

    pred_label  = model.predict(scaled)[0]
    pred_name   = le.inverse_transform([pred_label])[0]
    icon        = crop_icon(pred_name)

    # Probability distribution
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(scaled)[0]
        proba = np.exp(df_scores) / np.exp(df_scores).sum()
    else:
        proba = np.zeros(len(le.classes_))
        proba[pred_label] = 1.0

    top5_idx  = np.argsort(proba)[::-1][:5]
    top5_crops = [le.classes_[i] for i in top5_idx]
    top5_probs = [proba[i] for i in top5_idx]

    st.divider()

    #  Hero result card 
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg,#1a7f37,#2da44e);
                        border-radius:16px; padding:30px; text-align:center;
                        color:white; box-shadow:0 4px 15px rgba(0,0,0,0.2);">
                <div style="font-size:64px;">{icon}</div>
                <div style="font-size:26px; font-weight:700; margin-top:10px;">
                    {pred_name.title()}
                </div>
                <div style="font-size:14px; opacity:0.85; margin-top:6px;">
                    Recommended Crop
                </div>
                <div style="font-size:20px; font-weight:600; margin-top:10px;">
                    {top5_probs[0]*100:.1f}% confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Top 5 Predictions")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colours = ["#2da44e" if i == 0 else "#88c9a1" for i in range(len(top5_crops))]
        bars = ax.barh(
            [f"{crop_icon(c)}  {c.title()}" for c in top5_crops[::-1]],
            [p * 100 for p in top5_probs[::-1]],
            color=colours[::-1], edgecolor="white", height=0.55,
        )
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 105)
        ax.tick_params(axis="y", labelsize=10)
        for bar, val in zip(bars, top5_probs[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", fontsize=9, color="#333")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    #  Gauge-style confidence bar 
    st.divider()
    st.subheader("All Class Probabilities")
    all_crops = le.classes_
    all_probs  = proba

    fig2, ax2 = plt.subplots(figsize=(12, max(5, len(all_crops) * 0.38)))
    sorted_idx = np.argsort(all_probs)
    sorted_crops = [all_crops[i] for i in sorted_idx]
    sorted_probs = all_probs[sorted_idx]
    bar_colors = ["#2da44e" if c == pred_name else "#c6e6d1" for c in sorted_crops]

    ax2.barh(
        [f"{crop_icon(c)}  {c.title()}" for c in sorted_crops],
        sorted_probs * 100,
        color=bar_colors, edgecolor="white", height=0.7,
    )
    ax2.set_xlabel("Confidence (%)", fontsize=11)
    ax2.set_xlim(0, 105)
    ax2.tick_params(axis="y", labelsize=9)
    for i, (p, c) in enumerate(zip(sorted_probs, sorted_crops)):
        if p > 0.005:
            ax2.text(p * 100 + 0.3, i, f"{p*100:.1f}%", va="center", fontsize=8, color="#444")
    highlighted = mpatches.Patch(color="#2da44e", label=f"Recommended: {pred_name.title()}")
    ax2.legend(handles=[highlighted], loc="lower right", fontsize=10)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    #  Agronomic tips 
    st.divider()
    st.subheader(f"Growing Tips for {pred_name.title()}")
    TIPS = {
        "rice":       "Requires flooded or waterlogged soils. Best grown in humid tropical climates with heavy rainfall.",
        "maize":      "Thrives in well-drained loamy soil. Needs full sunlight and moderate rainfall (500–800 mm).",
        "banana":     "Grows best in deep, well-drained loam soils rich in organic matter. Needs high humidity.",
        "mango":      "Prefers deep, well-drained alluvial or loamy soils. Drought-tolerant once established.",
        "grapes":     "Best in deep, well-drained loamy to sandy loam soils. Sensitive to waterlogging.",
        "watermelon": "Thrives in sandy loam soils with good drainage. Requires warm temperatures and low humidity.",
        "coconut":    "Grows best in sandy loam soils near the coast. High tolerance for saline conditions.",
        "coffee":     "Requires well-drained, fertile volcanic soils. Grows best in highlands with consistent rainfall.",
        "cotton":     "Prefers deep, well-drained black or alluvial soils. Sensitive to waterlogging.",
        "potato":     "Requires loose, well-drained sandy loam soil. Cool climates with moderate rainfall.",
        "cashew":     "Grows on sandy coastal soils. Drought-resistant and suitable for marginal lands.",
        "palm tree":  "Thrives in deep, well-drained loamy soils in humid tropical regions with high rainfall.",
    }
    tip = TIPS.get(pred_name.lower(),
                   f"{pred_name.title()} grows best when soil nutrients, pH, and rainfall conditions "
                   f"are optimised as shown in your input parameters above.")
    st.info(f"🌿 **{pred_name.title()}:** {tip}")

else:
    #  Welcome state 
    st.info(
        "**Adjust the sliders** in the sidebar to match your farm's soil and climate "
        "conditions, then click **Predict Crop** to get a recommendation."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Crops", "22", "African crop types")
    with col2:
        st.metric("Model", "ML Classifier", "Trained on 2,200 samples")
    with col3:
        st.metric("Features", "6", "Soil & climate variables")

#  Footer 
st.divider()
st.caption(
    "🌾 Crop Prediction App by ALFA SIMON ENEKELE  |  Mat. No. 22L1CS0235 · Built with Streamlit · "
    "Model trained on the African Crop Recommendation Dataset"
)
