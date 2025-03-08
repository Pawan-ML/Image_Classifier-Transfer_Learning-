# Image Classifier Using Transfer Learning

## Description

![Project Image](cat_dog_classification.jpg)

This project implements an image classification model using transfer learning. A pre-trained deep learning model is fine-tuned to classify images of cats and dogs, significantly improving training efficiency and accuracy.

## Dataset Information
- Dataset: [Cats and Dogs for Classification](https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification)
- Training and test images are stored in separate directories.

## Features
- Uses Transfer Learning for improved classification performance
- Pre-trained model (e.g., MobileNetV2, VGG16, or ResNet) is used as a base
- Image preprocessing and augmentation for better generalization
- Model evaluation with accuracy and loss metrics

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Pandas
- OpenDatasets

## Installation & Setup
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib tensorflow opendatasets
   ```
2. Download the dataset using OpenDatasets:
   ```python
   import opendatasets as od
   od.download("https://www.kaggle.com/datasets/dineshpiyasamara/cats-and-dogs-for-classification")
   ```

## Model Training & Evaluation
1. Load dataset and preprocess images.
2. Use a pre-trained CNN model (e.g., MobileNetV2, ResNet, or VGG16).
3. Fine-tune the model by adding fully connected layers.
4. Train the model using training data.
5. Evaluate model performance on test data.
6. Visualize accuracy and loss over epochs.

## Usage
Run the Jupyter Notebook to train and evaluate the model:
```bash
jupyter notebook Image_Classifier(Transfer_Learning).ipynb
```

## Results
- Model accuracy and loss graphs
- Predictions on test images

## License
This project is for educational purposes only.

---
Developed by **Charutha Pawan Kodikara** 🚀

