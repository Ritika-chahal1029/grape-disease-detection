import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# load model
model = load_model("model/grape_model.h5")

# class names (same as training)
classes = ["black_rot", "esca", "leaf_blight", "healthy"]

# causes
causes = {
    "black_rot": "Fungal infection caused by warm and humid conditions.",
    "esca": "Fungal disease due to infected wood and poor vineyard management.",
    "leaf_blight": "Caused by fungal pathogens in moist environments.",
    "healthy": "No disease present."
}

# prevention
prevention = {
    "black_rot": "Remove infected leaves, ensure good air circulation, and apply fungicides.",
    "esca": "Prune infected branches and maintain vineyard hygiene.",
    "leaf_blight": "Avoid overhead watering and use proper fungicides.",
    "healthy": "Maintain proper care and regular monitoring."
}

# title
st.title("Grape Leaf Disease Detection 🍇")

# upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

# main logic
if uploaded_file is not None:
    # show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (128, 128))

    img = np.array(img)
    img = preprocess_input(img)   

    img = np.reshape(img, (1, 128, 128, 3))

    # prediction
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img)
        st.write("Prediction:", prediction)
        st.write("Index:", np.argmax(prediction))

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # output
    st.success(" Prediction: " + predicted_class)
    st.info(" Cause: " + causes[predicted_class])
    st.warning(" Prevention: " + prevention[predicted_class])
    st.write(" Confidence:", f"{confidence:.2f}%")

    # note
    st.caption(" Note: Model works only for grape leaves. Other images may give incorrect results.")