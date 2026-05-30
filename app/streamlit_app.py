"""
Streamlit app untuk PII Shield.
Demo aplikasi deteksi dan redaksi PII.
"""

import streamlit as st
import json
import sys
sys.path.insert(0, '/home/ata/school/fp-ml/src')

from data_loader import DataLoader


# Configure page
st.set_page_config(
    page_title="PII Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("# 🛡️ PII Shield: Deteksi dan Redaksi Data Pribadi")
st.markdown("**Sistem deteksi Personally Identifiable Information (PII) pada teks pendidikan**")

# Load label schema
@st.cache_resource
def load_label_schema():
    loader = DataLoader()
    return loader.label_schema


label_schema = load_label_schema()

# Sidebar
st.sidebar.markdown("## ⚙️ Konfigurasi")
model_choice = st.sidebar.selectbox(
    "Pilih model untuk deteksi:",
    ["Logistic Regression", "Linear SVM", "CRF", "XGBoost", "DistilBERT", "Ensemble"]
)

show_tokens = st.sidebar.checkbox("Tampilkan token details", value=False)
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Kontribusi")
st.sidebar.markdown("""
- **Qurrata A'yun Kamil** (5025241031) - Project Lead
- **Naura Taqiyya Mazi** (5025241119) - Baseline ML
- **Farras Nizar** (5025241143) - NER/CRF
- **Naufal Daffa Alfa Zain** (5025241066) - Deep Learning
- **Erica Triana** (5025241069) - Boosting Models
""")

# Main content
tab1, tab2, tab3 = st.tabs(["📄 Input Text", "📊 Hasil Deteksi", "ℹ️ Info"])

with tab1:
    st.markdown("## Masukkan teks untuk deteksi PII")
    
    input_choice = st.radio(
        "Pilih input method:",
        ["Manual input", "Contoh predefined"],
        horizontal=True
    )
    
    if input_choice == "Manual input":
        user_text = st.text_area(
            "Masukkan teks di sini:",
            height=200,
            placeholder="Contoh: My name is John Smith and my email is john@example.com"
        )
    else:
        examples = {
            "Contoh 1": "My name is Michael Jordan and my email is mjordan@nba.com",
            "Contoh 2": "You can contact me at sarah.johnson@school.edu or call 555-1234",
            "Contoh 3": "ID saya adalah 12345678 dan username saya adalah student_123",
        }
        selected_example = st.selectbox("Pilih contoh:", list(examples.keys()))
        user_text = examples[selected_example]
    
    if user_text:
        st.info(f"✓ Teks siap: {len(user_text)} karakter, ~{len(user_text.split())} kata")

with tab2:
    st.markdown("## Hasil Deteksi PII")
    
    if user_text:
        # Simple tokenization (untuk demo)
        tokens = user_text.split()
        
        # Dummy predictions (dalam praktik, gunakan model yang sebenarnya)
        # Untuk sekarang, ini hanya placeholder
        st.warning("⚠️ Model belum diimplementasi. Ini adalah interface demo.")
        st.markdown("""
        Ketika model siap diintegrasikan:
        1. Setiap token akan diklasifikasikan
        2. Hasil akan ditampilkan di bawah
        3. Teks akan di-highlight sesuai PII type
        4. Output redacted text tersedia untuk download
        """)
        
        # Example output
        st.markdown("### Contoh Output (jika model aktif)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📌 Token Classification")
            st.json({
                "tokens": tokens[:5] + (["..."] if len(tokens) > 5 else []),
                "labels": ["O", "B-NAME_STUDENT", "O", "B-EMAIL", "O"] + (["..."] if len(tokens) > 5 else [])
            })
        
        with col2:
            st.markdown("#### 🔐 PII Entities Detected")
            st.json({
                "entities": [
                    {"type": "NAME_STUDENT", "value": "[First Last]", "position": "0-2"},
                    {"type": "EMAIL", "value": "[email@example.com]", "position": "5-7"}
                ]
            })
        
        st.markdown("#### ✂️ Redacted Text")
        redacted_example = "My name is [NAME_STUDENT] and my email is [EMAIL]"
        st.text_area("Teks dengan redaksi:", value=redacted_example, height=100, disabled=True)
        
        # Download button
        st.download_button(
            label="⬇️ Download hasil (CSV)",
            data="document_id,token,true_label,pred_label\n1,My,O,O\n1,name,O,O",
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("📝 Masukkan teks terlebih dahulu di tab **Input Text**")

with tab3:
    st.markdown("## ℹ️ Informasi Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Dataset")
        st.markdown("""
        - **Sumber**: Kaggle Competition - PII Detection Removal from Educational Data
        - **Task**: Token Classification / Named Entity Recognition
        - **Format**: BIO tagging scheme
        """)
        
        st.markdown("### 🏷️ PII Labels")
        labels_list = [
            "NAME_STUDENT - Nama siswa",
            "EMAIL - Alamat email",
            "USERNAME - Username/akun",
            "ID_NUM - Nomor identitas",
            "PHONE_NUM - Nomor telepon",
            "URL_PERSONAL - URL personal",
            "STREET_ADDRESS - Alamat jalan",
        ]
        for label in labels_list:
            st.markdown(f"- `{label}`")
    
    with col2:
        st.markdown("### 📊 Model yang Digunakan")
        models = [
            "Logistic Regression (baseline)",
            "Linear SVM (baseline)",
            "CRF (sequence model)",
            "XGBoost / LightGBM (boosting)",
            "DistilBERT / DeBERTa (transformer)",
            "Hybrid Rule + ML",
            "Voting Ensemble",
        ]
        for i, model in enumerate(models, 1):
            st.markdown(f"{i}. {model}")
        
        st.markdown("### 📈 Metrik Utama")
        st.markdown("""
        - **Precision**: Akurasi positif predictions
        - **Recall**: Cakupan actual PII entities
        - **F1-Score**: Harmonic mean dari precision & recall
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 Link Penting")
    st.markdown("""
    - [Kaggle Competition](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
    - [Project README](./README.md)
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>PII Shield Project | Final Project Machine Learning | ITS 2026</small>
</div>
""", unsafe_allow_html=True)
