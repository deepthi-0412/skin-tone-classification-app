import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from skin_preprocess import extract_skin

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Skin Tone Classification System",
    page_icon="🧴",
    layout="wide"
)

# ================= HEADER =================
st.title("🧴 AI-Based Human Skin Tone Classification")
st.markdown("An intelligent system for automated skin tone detection using Deep Learning and Image Processing.")
st.markdown("---")

# ================= ABOUT SECTION =================
with st.expander("📌 About This System"):
    st.write("""
    This application performs:
    - Face Detection & Skin Region Extraction
    - LAB Color Space Conversion
    - CNN-based Skin Tone Classification
    - Skin Behavior & Care Recommendation

    The system is designed for automated and objective skin analysis.
    """)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_tone_model.h5")

model = load_model()

CLASS_NAMES = ["WHITE", "BROWN", "BLACK"]

# ================= SKIN BEHAVIOR INFO =================
def skin_behavior_info(predicted_class):
    info = {
        "WHITE": {
            "type": "Type I–II (Very Fair)",
            "burns": "Burns very easily",
            "tans": "Tans very slowly",
            "sensitivity": "Very high UV sensitivity",
            "care": "Use SPF 50+, avoid prolonged sun exposure"
        },
        "BROWN": {
            "type": "Type III–IV (Medium)",
            "burns": "May burn moderately",
            "tans": "Tans gradually",
            "sensitivity": "Moderate UV sensitivity",
            "care": "Use SPF 30+, limited sun exposure"
        },
        "BLACK": {
            "type": "Type V–VI (Dark)",
            "burns": "Rarely burns",
            "tans": "Tans easily and quickly",
            "sensitivity": "Low UV sensitivity",
            "care": "Use SPF 15+, keep skin moisturized"
        }
    }
    return info[predicted_class]

# ================= UI =================
st.subheader("📤 Upload Facial Image")
uploaded_file = st.file_uploader(
    "Upload a clear facial image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ================= LOAD IMAGE =================
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    # ================= LIGHTING CHECK (EXTRA FEATURE 🔥) =================
    brightness = np.mean(img_rgb)
    if brightness < 50:
        st.warning("⚠️ Image appears too dark. Better lighting may improve accuracy.")
    elif brightness > 200:
        st.warning("⚠️ Image appears too bright. Lighting may affect prediction.")

    # 🔴 IMPORTANT: Convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ================= SKIN EXTRACTION =================
    skin_bgr = extract_skin(img_bgr)

    # Convert back to RGB for display
    skin_display = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2RGB)

    # ================= PREPARE FOR MODEL =================
    lab = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2LAB)
    lab = cv2.resize(lab, (224, 224))
    lab = lab.astype("float32") / 255.0
    input_img = np.expand_dims(lab, axis=0)

    # ================= PREDICTION =================
    preds = model.predict(input_img)
    pred_index = np.argmax(preds)
    confidence = np.max(preds) * 100
    pred_class = CLASS_NAMES[pred_index]

    skin_info = skin_behavior_info(pred_class)

    st.markdown("---")

    # ================= DISPLAY IMAGES =================
    st.markdown("### 🖼️ Image Processing Stages")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(image, caption="Original Image", use_column_width=True)

    with c2:
        st.image(skin_display, caption="Extracted Skin Region", use_column_width=True)

    with c3:
        st.image(lab, caption="LAB Color Space Image", use_column_width=True)

    st.markdown("---")

    # ================= RESULTS =================
    st.markdown("### 📊 Prediction Result")
    st.success(f"✅ Predicted Skin Tone: **{pred_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    # ================= SKIN ANALYSIS =================
    st.markdown("### 🧬 Skin Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Skin Type:** {skin_info['type']}")
        st.write(f"**Burn Tendency:** {skin_info['burns']}")
        st.write(f"**Tanning Speed:** {skin_info['tans']}")

    with col2:
        st.write(f"**UV Sensitivity:** {skin_info['sensitivity']}")
        st.write(f"**Skin Care Recommendation:** {skin_info['care']}")

    # ================= CONFIDENCE GRAPH =================
    st.markdown("### 📊 Prediction Confidence Graph")

    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0] * 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Skin Tone Prediction Probabilities")

    st.pyplot(fig)

st.markdown("---")
st.markdown("© 2026 | B.Tech Final Year Project | AI Skin Tone Classification System")