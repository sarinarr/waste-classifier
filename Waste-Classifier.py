import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import os
import requests

weights_url = "https://huggingface.co/sarinarr/waste-classifier-weights/resolve/main/weights.hdf5"
weights_path = "weights.hdf5"

if not os.path.exists(weights_path):
    with st.spinner("Downloading model weights..."):
        with requests.get(weights_url, stream=True) as r:
            r.raise_for_status()
            with open(weights_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    st.success("Model weights downloaded successfully.")

# Constants
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@st.cache_resource
def load_model_cached():
    with open('network.json', 'r') as json_file:
        json_saved_model = json_file.read()
    network_loaded = tf.keras.models.model_from_json(json_saved_model)
    network_loaded.load_weights('weights.hdf5')
    network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return network_loaded

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    model = load_model_cached()
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return CLASS_NAMES[class_idx], confidence

def page_intro():
    st.title("♻️ Waste Classification")
    st.subheader("📊Business Case: The Problem")
    st.markdown("""
    By **2030**, the world is expected to generate over **2.6 billion tons of waste** every year.  
    Current waste management systems are inefficient and suffer from:

    - **Manual human sorting** that handles only 1–2 items per second
    - **High processing costs** of **$50–$100 per ton**
    - **Contamination and classification error rates** between **15–25%**

    These inefficiencies lead to:
    - ⚠️ **High contamination rates**
    - 💸 **Increased processing costs**
    - ♻️ **Reduced recycling efficiency**
    """)

    st.subheader("💡What Our Project Does")
    st.markdown("""
    To address these challenges, our team built an **AI-powered waste classifier** based on **ResNet50**.

    The app uses computer vision to classify household or industrial waste images into six key categories:

    - **Cardboard**
    - **Glass**
    - **Metal**
    - **Paper**
    - **Plastic**
    - **Trash**

    This makes sorting faster, cheaper, and more reliable.
    """)

    st.subheader("✅ How It Solves the Problem")
    st.markdown("""
    - **Automates classification**, eliminating the need for slow human sorting
    - **Improves accuracy**, reducing contamination and error rates
    - **Cuts costs** by optimizing the processing chain
    - **Boosts recycling efficiency**, contributing to sustainability goals

    In short, our solution leverages AI to make waste management **smarter, greener, and more scalable**.
    """)

def page_predict():
    st.title("Waste Classification")
    st.write("Upload a photo or take one with your webcam. The model will predict the category and give you eco information.")

    input_method = st.radio("Select input method:", ["📁 Upload from device", "📷 Take a photo"])
    img = None

    if input_method == "📁 Upload from device":
        uploaded = st.file_uploader("Upload an image", type=['jpg', 'png'])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)

    elif input_method == "📷 Take a photo":
        cam = st.camera_input("Take a photo")
        if cam:
            img = Image.open(cam)
            st.image(img, caption="Captured Photo", use_container_width=True)

    if img and st.button("🔍 Predict Waste Category"):
        label, confidence = predict(img)
        st.success(f"Prediction: **{label.upper()}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        waste_info = {
            "cardboard": {
                "impact": "Cardboard has a mixed environmental impact. While it's a bio-based, recyclable, and biodegradable material, its production and disposal can negatively affect the environment. Concerns include water pollution from pulping, methane production in landfills, and resource consumption. However, cardboard's recyclability and potential to reduce plastic use offer some environmental benefits.",
                "bin": "🟦Blue bin",
                "link": "https://www.gwp.co.uk/guides/how-is-cardboard-recycled/"
            },
            "glass": {
                "impact": "Glass production and disposal have both positive and negative environmental impacts. While glass is 100% recyclable and can be recycled endlessly, its production is energy-intensive and relies on finite resources, potentially leading to land degradation and biodiversity loss. The heavy weight of glass also contributes to higher transportation emissions.",
                "bin": "🟩Green bin",
                "link": "https://www.recyclenow.com/how-to-recycle/glass-recycling"
            },
            "metal": {
                "impact": "Metal production and use have significant and widespread impacts on the environment, including resource depletion, habitat destruction, pollution, and contributions to climate change. These impacts stem from mining, processing, and the eventual disposal of metal-containing products.",
                "bin": "🟨Yellow bin",
                "link": "https://www.anis-trend.com/recycling-metals-5-simple-steps/"
            },
            "paper": {
                "impact": "The environmental impact of paper production and use is multifaceted, encompassing deforestation, air and water pollution, and waste generation. Paper production relies heavily on trees, contributing to deforestation and habitat loss, while the manufacturing process releases pollutants into the air and water. Additionally, paper waste, including discarded paper and cardboard, makes up a significant portion of landfill waste, with the potential to decompose and release harmful greenhouse gases.",
                "bin": "🟦Blue bin",
                "link": "https://www.recyclenow.com/how-to-recycle/paper-recycling"
            },
            "plastic": {
                "impact": "Plastic pollution has far-reaching and detrimental effects on the environment, impacting ecosystems, wildlife, and even human health. It contributes to climate change, pollutes ecosystems with microplastics and toxic chemicals, and poses threats to marine life through ingestion and entanglement.",
                "bin": "🟨Yellow bin",
                "link": "https://www.recyclenow.com/how-to-recycle/plastic-recycling"
            },
            "trash": {
                "impact": "Non-recyclable trash significantly harms the environment, leading to pollution, habitat destruction, and contributing to climate change. Landfills overflow, releasing harmful gases and leaching toxins into soil and water. Plastic waste, in particular, breaks down into microplastics, polluting ecosystems and harming wildlife.",
                "bin": "🟫Brown bin", "link": "https://northlondonheatandpower.london/alternative-ways-treat-non-recyclable-waste"
                },
            }

        info = waste_info.get(label.lower())

        if info:
            st.markdown(f"### ♻️ Environmental Info for: {label.title()}")
            st.write(f"🌍 **Impact:** {info['impact']}")
            st.write(f"🗑️ **Recycle in:** {info['bin']}")
            st.markdown(f"[🔗 Learn more about {label.lower()} recycling]({info['link']})")

def page_about():
    st.title("👋 Meet the Team")

    for name, image in [
        ("Uxia Lojo", "Uxia.jpeg"),
        ("Joy Zhong", "Joy.jpeg"),
        ("James Alarde", "James.jpeg"),
        ("Maria Vazquez", "Maria.jpeg"),
        ("Sarina Ratnabhas", "Sarina.jpeg")
    ]:
        st.markdown(f"### {name}")
        st.image(image, width=180)

    st.markdown("""
    We are a team passionate about using AI to build meaningful, sustainable solutions that tackle real-world problems like waste management. 🌍
    """)

def main():
    page = st.sidebar.radio("Go to", ["Introduction", "Waste Prediction", "About Us"])

    if page == "Introduction":
        page_intro()
    elif page == "Waste Prediction":
        page_predict()
    elif page == "About Us":
        page_about()

if __name__ == "__main__":
    main()




