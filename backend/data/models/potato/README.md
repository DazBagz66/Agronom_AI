---
license: mit
language: en
tags:
- potato-disease
- image-classification
- tensorflow
- cnn
- keras
datasets:
- PlantVillage
metrics:
- accuracy
---

# Potato Disease Classifier

CNN model for classifying potato leaf diseases: Early Blight, Late Blight, and Healthy.

## Model Description

A Sequential CNN built with TensorFlow/Keras that classifies potato leaf images into three categories.

## Architecture

- Input preprocessing: Resizing(256,256) + Rescaling(1/255)
- Data augmentation: RandomFlip + RandomRotation (training only)
- 6x Conv2D(32-64 filters, 3x3, ReLU) + MaxPooling2D(2x2) blocks
- Flatten + Dense(64, ReLU) + Dense(3, Softmax)
- Total params: 183,747

## Classes

| Class | Description |
|-------|-------------|
| `Potato___Early_blight` | Early blight disease infection |
| `Potato___Late_blight`  | Late blight disease infection |
| `Potato___healthy`      | Healthy potato leaf |

## Performance

- Test Accuracy: ~99.6%
- Test Loss: ~0.009
- Validation Accuracy: ~99.5%
- Validation Loss: ~0.011

## Usage

```python
import keras
import numpy as np

model = keras.models.load_model("potato-disease-model.keras")

def predict(image_path):
    img = keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    return predicted_class, confidence
