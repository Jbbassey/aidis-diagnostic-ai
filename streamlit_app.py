import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Page config
st.set_page_config(
    page_title="AIDIS – Diagnostic Imaging AI Demo",
    page_icon="🧠",
    layout="centered"
)

# Sidebar info
st.sidebar.markdown("""
### 🧠 AIDIS – AI Imaging Demo  
Live prototype for AI-powered disease prediction from medical images.  
Built for the STEEM Grant.

🔗 [GitHub Repo](https://github.com/Jbbassey/aidis-diagnostic-ai)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About this demo")
st.sidebar.info("""
This prototype of AIDIS (AI Diagnostic Imaging System) uses deep learning 
to analyze diagnostic medical images and return predicted conditions 
with confidence levels.

This project is part of a STEEM Grant application.
""")

# Title
st.title("🧠 AIDIS – Diagnostic Imaging AI Demo")

st.markdown(
    "Upload a diagnostic image (e.g., chest X-ray), and the AI model will predict likely conditions."
)

# Use pretrained ResNet for now (you can swap in medical models like CheXNet later)
model = models.resnet18(pretrained=True)
model.eval()

# Updated real-world medical image classes
labels = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
]

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Upload section
uploaded_file = st.file_uploader("📷 Upload a diagnostic image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
            img_t = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_t)
                _, predicted = torch.max(output, 1)
                prediction = labels[predicted.item() % len(labels)]  # % to prevent index errors
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()] * 100

        st.success(f"✅ Predicted: **{prediction}** ({confidence:.2f}% confidence)")
