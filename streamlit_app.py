import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import urllib

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

# Title
st.title("🧠 AIDIS – Diagnostic Imaging AI Demo")

st.markdown(
    "Upload a diagnostic image (e.g., chest X-ray), and the AI model will predict possible conditions."
)

# Load pretrained model (simplified ResNet)
model = models.resnet18(pretrained=True)
model.eval()

# Dummy labels for demo (you can replace with real medical classes)
labels = ['Normal', 'Pneumonia', 'Tuberculosis', 'No Finding']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Upload section
uploaded_file = st.file_uploader("Upload a diagnostic image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Analyze Image"):
        st.write("Analyzing...")

        img_t = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_t)
            _, predicted = torch.max(output, 1)
            prediction = labels[predicted.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()] * 100

        st.success(f"✅ Predicted: **{prediction}** ({confidence:.2f}% confidence)")
