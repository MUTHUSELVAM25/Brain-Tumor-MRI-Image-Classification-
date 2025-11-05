# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

# ----- Header -----
st.markdown("""
    <h1 style='text-align: center; color: #003366; font-size: 2.5em;'>🧠 Brain Tumor MRI Classifier</h1>
    <hr style='border: 2px solid #003366;'>
    <p style='font-size: 1.2em; color: #444; text-align: center;'>
        Upload a brain MRI image and instantly get a deep learning–powered tumor type prediction.
    </p>
""", unsafe_allow_html=True)

# Load your trained model
@st.cache_resource
def load_brain_model():
    return load_model("best_tl_model.h5")

model = load_brain_model()

# Class Labels - set correct order by model output
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

uploaded_file = st.file_uploader("Choose an MRI image (.jpg, .png)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_expanded = np.expand_dims(img_array, axis=0)

    st.image(img, caption='Uploaded MRI Image', width=300, use_column_width=False, clamp=True)
    st.info('Processing image...')

    # Model Prediction
    pred_probs = model.predict(img_expanded)[0]
    pred_index = np.argmax(pred_probs)
    pred_class = class_names[pred_index]
    confidence = pred_probs[pred_index]

    # ----- Output Section -----
    st.success(f"**Prediction:** {pred_class.upper()}")
    st.write(f"**Model Confidence:** {confidence:.2%}")

    # Show confidence as a progress bar
    st.progress(float(confidence))

    # ----- Show Probability Bar Chart -----
    fig, ax = plt.subplots(figsize=(5,2))
    plt.barh(class_names, pred_probs, color='#3CB371')
    plt.xlabel("Probability")
    plt.title("Class Probabilities")
    plt.xlim([0, 1])
    for i, v in enumerate(pred_probs):
        plt.text(v+0.01, i, f"{v:.2%}", color='black', fontweight='bold')
    st.pyplot(fig)

    # Class description (optional)
    tumor_info = {
        'glioma': "Gliomas are tumors that originate in the glial cells of the brain.",
        'meningioma': "Meningiomas typically arise from the membranes around the brain and spinal cord.",
        'no_tumor': "No visible tumor detected in this MRI scan.",
        'pituitary': "Pituitary tumors occur in the pituitary gland, affecting hormone production."
    }
    st.markdown(f"**About:** {tumor_info.get(pred_class, '')}")

else:
    st.warning("Please upload an MRI image to get prediction results.")

# Add footer
st.markdown("<hr><center>Made with ❤️ by MUTHU SELVAM | Powered by MobileNetV2 Deep Learning</center>", unsafe_allow_html=True)
