import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Aerial Image Classifier",
    page_icon="🛩️",
    layout="wide"
)

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    cnn_model = load_model("models/custom_cnn_best.h5", compile=False)
    tl_model = load_model("models/transfer_learning_best.h5", compile=False)
    
    # YOLO model
    yolo_model = YOLO("models/yolo_model.pt")  # change if needed
    
    return cnn_model, tl_model, yolo_model

# ---------------- PREPROCESS ---------------- #
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- CNN/TL PREDICTION ---------------- #
def predict_classification(model, image):
    processed = preprocess_image(image)
    pred = model.predict(processed)

    class_id = np.argmax(pred)
    confidence = float(np.max(pred))

    classes = ["Drone", "Bird"]
    return classes[class_id], confidence, pred

# ---------------- YOLO PREDICTION ---------------- #
def predict_yolo(model, image):
    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            detections.append((label, conf))

    return detections, results

# ---------------- MAIN APP ---------------- #
def main():
    st.title("🛩️ Aerial Image Classification & Detection System")
    st.markdown("### Detect **Drone or Bird** using CNN / Transfer Learning / YOLO")

    cnn_model, tl_model, yolo_model = load_models()

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    mode = st.sidebar.selectbox(
        "Choose Model",
        ["Custom CNN", "Transfer Learning", "YOLO Detection", "Compare All"]
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")  # ✅ FIXED

        if st.button("🔍 Predict"):

            with col2:
                st.subheader("🎯 Results")

                # -------- CNN -------- #
                if mode == "Custom CNN":
                    label, conf, raw = predict_classification(cnn_model, image)

                    st.success("Model: Custom CNN")
                    st.write(f"Prediction: **{label}**")
                    st.write(f"Confidence: **{conf:.2%}**")

                # -------- TRANSFER -------- #
                elif mode == "Transfer Learning":
                    label, conf, raw = predict_classification(tl_model, image)

                    st.success("Model: Transfer Learning")
                    st.write(f"Prediction: **{label}**")
                    st.write(f"Confidence: **{conf:.2%}**")

                # -------- YOLO -------- #
                elif mode == "YOLO Detection":
                    detections, results = predict_yolo(yolo_model, image)

                    st.success("Model: YOLOv8")

                    if detections:
                        for label, conf in detections:
                            st.write(f"{label} ({conf:.2%})")
                    else:
                        st.warning("No objects detected")

                    # Show annotated image
                    annotated = results[0].plot()
                    st.image(annotated, caption="YOLO Detection", width="stretch")

                # -------- COMPARE ALL -------- #
                else:
                    st.success("🔍 Model Comparison")

                    col_a, col_b, col_c = st.columns(3)

                    # CNN
                    with col_a:
                        label, conf, _ = predict_classification(cnn_model, image)
                        st.markdown("### 🧠 CNN")
                        st.write(label, f"({conf:.2%})")

                    # TL
                    with col_b:
                        label, conf, _ = predict_classification(tl_model, image)
                        st.markdown("### ⚡ Transfer")
                        st.write(label, f"({conf:.2%})")

                    # YOLO
                    with col_c:
                        detections, _ = predict_yolo(yolo_model, image)
                        st.markdown("### 🎯 YOLO")

                        if detections:
                            for label, conf in detections:
                                st.write(label, f"({conf:.2%})")
                        else:
                            st.write("No detection")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()