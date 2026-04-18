# grape-disease-detection
Grape Leaf Disease Detection using CNN (EfficientNet) with Streamlit Deployment

##  Overview

This project is a deep learning-based web application that detects diseases in grape leaves using image classification. The model is trained using a Convolutional Neural Network (EfficientNetB0) and deployed with Streamlit.

##  Features

* Upload grape leaf image
* Predict disease category
* Display prediction confidence
* Show causes and prevention methods

##  Model Details

* Architecture: CNN (EfficientNetB0)
* Accuracy: ~98%
* Classes:

  * Black Rot
  * Esca
  * Leaf Blight
  * Healthy

##  Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy

##  Dataset

* Custom dataset collected and cleaned from multiple sources
* Handled class imbalance and real-world variations

##  How to Run

```bash
git clone https://github.com/Ritika-chahal1029/grape-disease-detection.git
cd grape-disease-detection
pip install -r requirements.txt
streamlit run app/app.py
```

##  Demo

![App Demo](image/demo.png)

##  Learnings

* Solved class imbalance problem using class weights
* Improved model performance through data cleaning
* Handled real-world image variations
* Built and deployed a complete ML pipeline


##  Author

Ritika Chahal

