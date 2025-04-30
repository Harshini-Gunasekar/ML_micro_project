import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("fabric_model.h5")

# Class names (same order as during training)
classes = ['aluminium_foil', 'brown_bread', 'corduroy', 'cotton', 'cracker',
           'linen', 'orange_peel', 'sandpaper', 'sponge', 'styrofoam']

