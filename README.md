# 🧵 Fabric Classification using EfficientNetV2S

This project classifies grayscale fabric textures from the [KTH-TIPS dataset](https://www.nada.kth.se/cvap/databases/kth-tips/) into 10 material categories using a fine-tuned EfficientNetV2S deep learning model.

## 📂 Dataset

The dataset consists of grayscale images from 10 fabric/material categories:

- aluminium_foil
- brown_bread
- corduroy
- cotton
- cracker
- linen
- orange_peel
- sandpaper
- sponge
- styrofoam

Each image is resized to `224x224`, normalized, and converted from grayscale to 3-channel RGB format for compatibility with EfficientNet.

## 🧠 Model

The model uses:

- **EfficientNetV2S** (pretrained on ImageNet, fine-tuned on KTH-TIPS)
- **Data Augmentation** (Random Flip, Rotation, Zoom)
- **Global Average Pooling + Dense layers**

Compiled with:
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Metrics: `Accuracy`

## 🏋️ Training

- 50 epochs
- Early stopping and learning rate reduction callbacks used
- Training, validation, and test split: 70/15/15
- Final model saved as `fabric_model.h5`

## 📈 Results

Plots of training/validation accuracy and loss are generated at the end of training.

## 🖼️ Prediction on Custom Image

To predict a single image:

```python
from tensorflow.keras.models import load_model
import cv2, numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = load_model("fabric_model.h5")

# Class names
classes = ['aluminium_foil', 'brown_bread', 'corduroy', 'cotton', 'cracker',
           'linen', 'orange_peel', 'sandpaper', 'sponge', 'styrofoam']

# Predict and visualize
def predict_and_show_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = np.stack([img]*3, axis=-1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    
    plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_class}")
    plt.axis("off")
    plt.show()

# Example usage
predict_and_show_image("path/to/your/image.png")
