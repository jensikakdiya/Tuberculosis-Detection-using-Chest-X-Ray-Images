# streamlit_run.py
import os
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np
import streamlit as st # type: ignore
import tensorflow as tf

st.set_page_config(page_title="TB Chest X-ray Classifier", page_icon="🩺", layout="centered")

# --- Helpers ---
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Normal", "TB"]

@st.cache_resource
def load_model_from_path(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image_pil(pil_img, img_size=IMG_SIZE):
    """
    Convert PIL image to grayscale, resize, scale to [0,1],
    and return shape (1, H, W, 1) numpy array for the model.
    """
    # convert to grayscale
    img = ImageOps.grayscale(pil_img)
    # resize
    img = img.resize(img_size)
    arr = np.asarray(img, dtype="float32") / 255.0
    # ensure shape H,W -> H,W,1 and expand batch
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(model, pil_img):
    x = preprocess_image_pil(pil_img)
    preds = model.predict(x)
    # binary sigmoid output expected -> preds shape (1,1) or (1,)
    prob = float(preds.ravel()[0])
    label = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    return label, prob

# --- UI ---
st.title("🩺 TB Chest X-ray Classifier")
st.markdown(
    """
Upload a chest X-ray image (PNG / JPG).  
Choose or upload a trained Keras .h5 model, then press *TEST* to classify the image as *Normal* or *TB*.
"""
)

with st.sidebar:
    st.header("Model options")
    st.markdown("Either pick a model file from this folder or upload your own .h5 model.")

    # list .h5 files in current directory
    h5_files = [f for f in os.listdir(".") if f.lower().endswith(".h5")]
    selected_model = None
    if h5_files:
        selected_model = st.selectbox("Choose model from working dir", ["-- select --"] + h5_files)
        if selected_model == "-- select --":
            selected_model = None

    uploaded_model_file = st.file_uploader("Or upload model (.h5)", type=["h5"])
    if uploaded_model_file is not None:
        # save uploaded model temporarily to load it
        tmp_model_path = os.path.join("uploaded_model.h5")
        with open(tmp_model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        selected_model = tmp_model_path
        st.success("Uploaded model saved as uploaded_model.h5")

    st.markdown("---")
    st.markdown("Model notes: the model should accept single-channel grayscale images of size 128×128 and output a single sigmoid probability (0..1).")
    st.caption("If you trained models with different input size, change IMG_SIZE in the script accordingly.")

# Main upload area
uploaded_img = st.file_uploader("Upload Chest X-ray image (PNG/JPG)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns([1, 2])
with col1:
    st.write("Preview")
    if uploaded_img is None:
        st.info("Upload an image to preview here.")
    else:
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, use_column_width=True)
        except Exception as e:
            st.error(f"Cannot open image: {e}")

with col2:
    st.write("Prediction")
    st.write("")
    # Action button
    run_test = st.button("✅ TEST")

# When TEST pressed
if run_test:
    if uploaded_img is None:
        st.warning("Please upload an X-ray image before pressing TEST.")
    else:
        # determine model path
        if selected_model is None:
            st.warning("Please select or upload a model (.h5) in the sidebar.")
        else:
            with st.spinner("Loading model..."):
                model = load_model_from_path(selected_model)
            if model is None:
                st.error("Model couldn't be loaded. Check the file and try again.")
            else:
                # prepare image
                try:
                    # If uploaded_img is BytesIO/UploadedFile, re-open
                    uploaded_img.seek(0)
                    pil_img = Image.open(uploaded_img)
                except Exception:
                    st.error("Failed to read uploaded image.")
                    pil_img = None

                if pil_img is not None:
                    with st.spinner("Running prediction..."):
                        try:
                            label, prob = predict(model, pil_img)
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            label, prob = None, None

                    if label is not None:
                        # show results
                        st.metric(label="Predicted", value=label, delta=f"{prob*100:.2f}%")
                        st.write(f"*Probability (TB)*: {prob:.4f}")
                        # interpretation
                        if prob >= 0.9:
                            st.success("High confidence — strongly suggests TB." if label == "TB" else "High confidence — strongly suggests Normal.")
                        elif prob >= 0.7:
                            st.info("Moderate confidence.")
                        else:
                            st.warning("Low confidence — consider further review or more data.")
                        # show preprocessed image used for model (grayscale resized)
                        st.markdown("*Preprocessed image fed to the model:*")
                        pre_img = preprocess_image_pil(pil_img)[0, :, :, 0]  # H,W
                        # scale back to 0-255 for display
                        display_img = Image.fromarray((pre_img * 255).astype("uint8"))
                        st.image(display_img, caption=f"Resized to {IMG_SIZE[0]}×{IMG_SIZE[1]} (grayscale)", use_column_width=False)
                        # Option to download result as text
                        result_txt = f"Prediction: {label}\nProbability(TB): {prob:.6f}\nModel: {selected_model}\n"
                        st.download_button("Download result", result_txt, file_name="prediction_result.txt")