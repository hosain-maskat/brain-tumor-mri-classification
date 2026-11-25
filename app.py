"""
Brain Tumor MRI Classification Web Application
Powered by Streamlit & Keras (Xception Architecture)
"""

import os
import sys

# Ensure Keras selects a compatible backend on Windows
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CONSTANTS & METADATA
# ---------------------------------------------------------
MODEL_CANDIDATE_PATHS = [
    "brain_tumor_model.keras",
    "My_Model2.keras",
    "model.keras",
    "brain_tumor_model.h5",
]

TARGET_IMAGE_SIZE = (299, 299)  # Height, Width matching Xception architecture
RESCALE_FACTOR = 1.0 / 255.0     # Rescale matching ImageDataGenerator(rescale=1/255)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_DETAILS: Dict[str, Dict[str, Any]] = {
    "glioma": {
        "display_name": "Glioma Tumor",
        "category": "Intracranial Neoplasm",
        "color": "#e03131",
        "badge_class": "badge-danger",
        "description": (
            "Gliomas originate in glial cells (astrocytes, oligodendrocytes) that support neurons in the brain. "
            "They represent one of the most common primary brain neoplasms, ranging from low-grade lesions to "
            "aggressive glioblastomas requiring immediate multimodal neuro-oncological care."
        ),
        "guidance": "Recommended clinical pathway: Comprehensive contrast-enhanced MRI (T1+Gad, T2, FLAIR) and urgent neurosurgical consultation."
    },
    "meningioma": {
        "display_name": "Meningioma Tumor",
        "category": "Meningeal Neoplasm",
        "color": "#f08c00",
        "badge_class": "badge-warning",
        "description": (
            "Meningiomas arise from the meninges (the protective membranes enveloping the brain and spinal cord). "
            "The vast majority are histologically benign and slow-growing, though symptoms arise from compression "
            "of adjacent brain parenchyma and cranial nerves."
        ),
        "guidance": "Recommended clinical pathway: Assessment of dural attachment, mass effect, perilesional edema, and monitoring via serial neuroimaging."
    },
    "notumor": {
        "display_name": "No Tumor Detected",
        "category": "Normal / Non-Tumorous Finding",
        "color": "#2f9e44",
        "badge_class": "badge-success",
        "description": (
            "No morphological imaging patterns indicative of glioma, meningioma, or pituitary adenoma "
            "were identified by the deep neural network on this evaluated MRI scan slice."
        ),
        "guidance": "Notice: An automated negative screening does not rule out non-neoplastic pathologies, micro-lesions, or anomalies in unexamined slices."
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "category": "Sellar / Pituitary Adenoma",
        "color": "#1971c2",
        "badge_class": "badge-info",
        "description": (
            "Pituitary adenomas originate within the pituitary gland situated in the sella turcica. "
            "Although almost always benign, they can produce hormonal dysregulation (hyper/hypopituitarism) "
            "or compress the adjacent optic chiasm, leading to visual field deficits."
        ),
        "guidance": "Recommended clinical pathway: Thin-slice sellar MRI with dynamic contrast enhancement, neuro-ophthalmological visual field exam, and endocrine panel."
    },
}

# ---------------------------------------------------------
# CUSTOM CSS STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 60%, #203a43 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #ccd9e8;
        font-size: 1.05rem;
    }
    
    /* Research Attribution Banner */
    .research-banner {
        background-color: #f0f7ff;
        color: #1c3d5a;
        border-left: 6px solid #1971c2;
        padding: 1.1rem 1.35rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.94rem;
        line-height: 1.6;
        border-top: 1px solid #d0ebff;
        border-right: 1px solid #d0ebff;
        border-bottom: 1px solid #d0ebff;
    }
    .research-banner strong {
        color: #1864ab;
    }

    /* Result Card */
    .result-container {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid #dee2e6;
        background: #ffffff;
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .result-container {
            background: #1e293b;
            border-color: #334155;
            color: #f8fafc;
        }
        .research-banner {
            background-color: #102a43;
            color: #d0ebff;
            border-color: #1864ab;
            border-left-color: #339af0;
        }
        .research-banner strong {
            color: #74c0fc;
        }
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
def find_model_path() -> Optional[str]:
    """Scan directory for candidate model files."""
    for filename in MODEL_CANDIDATE_PATHS:
        if os.path.isfile(filename):
            return filename
    return None


@st.cache_resource(show_spinner="Initializing neural network model...")
def load_classification_model(model_filepath: str):
    """
    Loads Keras model with compilation disabled for fast inference.
    Supports Keras 3 with PyTorch or TensorFlow backends.
    """
    # Attempt 1: Keras 3
    try:
        import keras
        model = keras.models.load_model(model_filepath, compile=False)
        backend = getattr(keras.backend, "backend", lambda: "native")()
        return model, f"Keras 3 ({backend} backend)"
    except Exception as e_keras:
        # Attempt 2: TensorFlow Keras
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_filepath, compile=False)
            return model, f"TensorFlow {tf.__version__}"
        except Exception as e_tf:
            st.error(f"Error loading model from `{model_filepath}`.\nKeras error: {e_keras}\nTF error: {e_tf}")
            raise e_keras


# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess image to match training pipeline:
    1. Color space: 3-channel RGB (handles Grayscale, RGBA, etc.)
    2. Resolution: (299, 299) using bilinear interpolation
    3. Normalization: Rescale pixel values from [0, 255] to [0.0, 1.0]
    4. Batch dimension: Shape (1, 299, 299, 3)
    """
    orig_size = pil_img.size  # (width, height)
    orig_mode = pil_img.mode

    # Step 1: Ensure RGB
    if pil_img.mode != "RGB":
        rgb_img = pil_img.convert("RGB")
    else:
        rgb_img = pil_img

    # Step 2: Resize
    resized_img = rgb_img.resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)

    # Step 3: Rescale
    img_array = np.array(resized_img, dtype=np.float32)
    img_rescaled = img_array * RESCALE_FACTOR

    # Step 4: Batch dimension
    batch_tensor = np.expand_dims(img_rescaled, axis=0)

    metadata = {
        "original_resolution": f"{orig_size[0]} x {orig_size[1]} px",
        "original_format": orig_mode,
        "preprocessed_shape": list(batch_tensor.shape),
        "target_resolution": f"{TARGET_IMAGE_SIZE[0]} x {TARGET_IMAGE_SIZE[1]} px",
        "pixel_intensity_range": f"[{img_rescaled.min():.2f}, {img_rescaled.max():.2f}]",
        "rescale_factor": f"{RESCALE_FACTOR:.6f} (1 / 255.0)",
    }

    return batch_tensor, metadata


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
def render_sidebar(model_path: Optional[str], backend_name: Optional[str]):
    with st.sidebar:
        st.title("System Information")
        
        st.subheader("Model Specifications")
        if model_path:
            st.success(f"**Loaded Model:** `{os.path.basename(model_path)}`")
            st.markdown(f"- **Engine:** {backend_name or 'Ready'}")
            st.markdown(f"- **Architecture:** Xception + MLP Head")
            st.markdown(f"- **Input Dimensions:** `(299, 299, 3)`")
            st.markdown(f"- **Pixel Scaling:** `1 / 255.0` (`[0, 1]`)")
            st.markdown(f"- **Target Classes:** 4")
        else:
            st.error("No compatible `.keras` model found.")
            st.info("Place `brain_tumor_model.keras` in the project root directory.")

        st.markdown("---")
        st.subheader("Target Pathologies")
        for key, info in CLASS_DETAILS.items():
            st.markdown(f"**{info['display_name']}**")
            st.caption(info["category"])

        st.markdown("---")
        st.caption("NeuroScan AI | Research & Educational Prototype")


# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
def main():
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <h1>🧠 NeuroScan AI - Brain MRI Tumor Classification</h1>
            <p>Automated deep learning analysis of cranial MRI scans for Glioma, Meningioma, Pituitary tumor, or No Tumor findings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Research Attribution & Academic Guidance Banner
    st.markdown(
        """
        <div class="research-banner">
            <strong>🎓 Research Attribution & Academic Guidance:</strong><br>
            This research was conducted under the leading supervision and academic guidance of <strong>Md. Ali Hossain</strong>, 
            whose significant contributions were instrumental in the successful development of this deep learning framework. 
            <strong>Hosain Mohammad Maskat</strong> carried out the primary implementation, model development, and experimental work under his guidance, 
            with valuable assistance from <strong>Tanbir Hasan Jewel</strong>. Furthermore, <strong>Ahmed Moustafa</strong>, 
            <strong>Md. Shofiqul Islam</strong>, <strong>Mayen Uddin Mojumdar</strong>, and <strong>Md. Joy Mamun</strong> provided 
            valuable academic, technical, and research contributions throughout the project. Their collective efforts and guidance 
            facilitated the development of an Xception-based deep learning model for brain tumor MRI classification, achieving promising predictive performance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Locate Model
    model_path = find_model_path()
    if not model_path:
        st.error(
            "⚠️ **Model file not found.** Please ensure `brain_tumor_model.keras` or `My_Model2.keras` "
            "is located in the project folder."
        )
        st.stop()

    try:
        model, backend_info = load_classification_model(model_path)
    except Exception as e:
        st.error(f"Failed to initialize model: {e}")
        st.stop()

    render_sidebar(model_path, backend_info)

    # Check for sample images
    sample_dir = Path("sample_images")
    sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) if sample_dir.is_dir() else []

    # File Upload & Sample Image Selection
    col_input, col_specs = st.columns([2, 1], gap="medium")

    selected_image = None
    image_source_label = ""

    with col_input:
        st.subheader("1. Input Brain MRI Scan")
        
        tab_upload, tab_sample = st.tabs(["📁 Upload Image File", "🧪 Choose Sample MRI Scan"])
        
        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload a cranial MRI image (JPG, PNG, JPEG, BMP)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Accepts axial, coronal, or sagittal T1/T2/FLAIR cranial MRI slices.",
            )
            if uploaded_file is not None:
                try:
                    selected_image = Image.open(uploaded_file)
                    image_source_label = f"Uploaded file: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")

        with tab_sample:
            if sample_files:
                sample_options = {f.stem.replace("_", " ").title(): f for f in sample_files}
                chosen_sample = st.selectbox("Select a pre-loaded sample MRI:", list(sample_options.keys()))
                if st.button("Load Selected Sample"):
                    sample_path = sample_options[chosen_sample]
                    selected_image = Image.open(sample_path)
                    image_source_label = f"Sample: {chosen_sample}"
                    st.session_state["sample_loaded"] = sample_path
                elif "sample_loaded" in st.session_state and uploaded_file is None:
                    selected_image = Image.open(st.session_state["sample_loaded"])
                    image_source_label = f"Sample: {Path(st.session_state['sample_loaded']).stem.replace('_', ' ').title()}"
            else:
                st.caption("No local sample images found in `sample_images/` directory.")

    with col_specs:
        st.subheader("Preprocessing Pipeline")
        st.markdown(
            """
            - **Target Resolution:** `299 x 299` px (Bilinear)
            - **Color Space:** 3-channel RGB
            - **Pixel Scaling:** `1 / 255.0` (`[0.0, 1.0]`)
            - **Architecture:** Xception (Pretrained ImageNet)
            - **Classifier Output:** 4-Class Softmax
            """
        )

    if selected_image is None:
        st.info("👆 Upload an MRI image file or select a sample scan above to generate a classification.")
        return

    # Image Inspection Section
    st.markdown("---")
    st.subheader("2. Image Inspection & Preprocessing Verification")

    tensor_input, meta = preprocess_image(selected_image)

    col_orig, col_prep, col_meta = st.columns([1, 1, 1], gap="medium")

    with col_orig:
        st.markdown(f"**Source Image** ({image_source_label})")
        st.image(selected_image, use_container_width=True, caption=f"Original Size: {meta['original_resolution']}")

    with col_prep:
        st.markdown("**Preprocessed Input (299 x 299)**")
        preview_img = selected_image.convert("RGB").resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
        st.image(preview_img, use_container_width=True, caption=f"Input Tensor: {meta['target_resolution']}")

    with col_meta:
        st.markdown("**Preprocessing Specifications**")
        st.json(meta)

    # Model Inference
    st.markdown("---")
    st.subheader("3. Model Prediction & Diagnostic Analysis")

    with st.spinner("Executing neural network forward pass..."):
        start_time = time.perf_counter()
        raw_predictions = model.predict(tensor_input, verbose=0)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000

    probabilities = raw_predictions[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    predicted_info = CLASS_DETAILS[predicted_class]
    confidence_pct = probabilities[predicted_idx] * 100.0

    # Primary Result Metrics
    res_col1, res_col2, res_col3 = st.columns([2, 1, 1], gap="medium")

    with res_col1:
        st.markdown(
            f"""
            <div style="background-color: {predicted_info['color']}15; border-left: 6px solid {predicted_info['color']}; padding: 1.25rem; border-radius: 8px;">
                <div class="metric-label">Predicted Finding</div>
                <div class="metric-value" style="color: {predicted_info['color']};">
                    {predicted_info['display_name']}
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.95rem; color: #495057;">
                    <strong>Pathology Category:</strong> {predicted_info['category']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.metric(
            label="Confidence Score",
            value=f"{confidence_pct:.2f}%",
            delta="High Certainty" if confidence_pct >= 85.0 else ("Moderate" if confidence_pct >= 60.0 else "Low Certainty"),
        )

    with res_col3:
        st.metric(
            label="Inference Latency",
            value=f"{elapsed_time_ms:.1f} ms",
            delta="Optimized",
        )

    # Clinical Context Card
    st.markdown("#### Clinical Background")
    st.info(f"**Overview:** {predicted_info['description']}")
    st.caption(f"📋 **Clinical Guidance:** {predicted_info['guidance']}")

    # Probability Chart
    st.markdown("#### Multiclass Probability Breakdown")

    df_probs = pd.DataFrame({
        "Condition": [CLASS_DETAILS[c]["display_name"] for c in CLASS_NAMES],
        "Probability (%)": [round(float(p) * 100.0, 2) for p in probabilities],
        "Raw Score": [float(p) for p in probabilities],
        "Class Key": CLASS_NAMES,
    })

    df_sorted = df_probs.sort_values(by="Probability (%)", ascending=True)

    fig = px.bar(
        df_sorted,
        x="Probability (%)",
        y="Condition",
        orientation="h",
        text="Probability (%)",
        color="Condition",
        color_discrete_map={
            CLASS_DETAILS[c]["display_name"]: CLASS_DETAILS[c]["color"]
            for c in CLASS_NAMES
        },
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
    )

    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Probability Score (%)", showgrid=True),
        yaxis=dict(title=""),
        height=280,
        margin=dict(l=10, r=20, t=10, b=10),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Probability Table
    with st.expander("📊 View Detailed Probability Scores"):
        st.dataframe(
            df_probs[["Condition", "Probability (%)", "Raw Score"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
