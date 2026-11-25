# 🧠 NeuroScan AI - Brain Tumor MRI Classification Web App

A clean, production-ready Streamlit web application for brain tumor classification using deep learning (Xception model trained on cranial MRI scans).

---

## 🎯 Features

- **MRI Image Upload**: Supports JPG, PNG, JPEG, BMP cranial MRI images.
- **Sample Scans Included**: Includes pre-loaded sample MRI scans (`sample_images/`) for instant testing without needing your own files.
- **Training-Matched Preprocessing**:
  - Image resizing to **299 × 299 px** using bilinear interpolation.
  - Automatic color space conversion to **3-channel RGB**.
  - Normalization: pixel rescaling by **1/255.0** to range `[0.0, 1.0]`.
  - Batch expansion to tensor shape `(1, 299, 299, 3)`.
- **Accurate Model Inference**:
  - Caches model in memory (`@st.cache_resource`) for sub-second repeat predictions.
  - Multi-backend support (Keras 3 with PyTorch/TensorFlow backend).
- **Comprehensive Clinical & Visual Output**:
  - Predicted finding badge with color-coded classification.
  - Confidence percentage and inference latency.
  - Interactive horizontal probability distribution bar chart (Plotly).
  - Detailed pathology description and clinical next steps.
  - Side-by-side inspection of original scan vs. 299×299 preprocessed tensor.
- **Prominent Medical Disclaimer**: Clear notice that the app is an educational/experimental research tool and not a clinical diagnostic device.

---

## 📁 Project Structure

```
Brain_tumor classification/
│
├── app.py                     # Main Streamlit web application
├── requirements.txt           # Python dependencies
├── run_app.bat                # One-click Windows launch script
├── brain_tumor_model.keras    # Deep learning model (Xception)
├── My_Model2.keras            # Original model archive
│
├── sample_images/             # Sample MRI scans for testing
│   ├── sample_glioma.jpg
│   ├── sample_meningioma.jpg
│   └── sample_pituitary.jpg
│
└── README.md                  # Instructions and documentation
```

---

## 🔬 Classification Categories

| Class | Label | Description |
|---|---|---|
| **0** | `glioma` | Primary intra-axial tumor originating from glial cells |
| **1** | `meningioma` | Extra-axial tumor arising from the meninges |
| **2** | `notumor` | Healthy scan / no tumor detected in this slice |
| **3** | `pituitary` | Sellar / pituitary gland neoplasm |

---

## 🚀 How to Run on Windows

### Option 1: Quick One-Click Launch (Recommended)
Double-click `run_app.bat` in File Explorer.

---

### Option 2: Run via Terminal (PowerShell or Command Prompt)

1. Open PowerShell or Command Prompt in this folder:
   ```powershell
   cd "c:\Users\hosai\Desktop\Brain_tumor classification"
   ```

2. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
   *(If creating a new virtual environment from scratch)*:
   ```powershell
   uv venv .venv --python 3.12
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Launch the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

4. The app will open in your default browser at:
   ```
   http://localhost:8501
   ```

---

## ⚠️ Medical Disclaimer

> **DISCLAIMER:** This application is strictly an experimental research prototype for demonstration, education, and academic evaluation. It is **NOT** a certified medical diagnostic device and **MUST NOT** be used to make clinical treatment decisions or replace evaluation by licensed radiologists and neurosurgeons.
