# Image Classification Using Convolutional Neural Networks (CNN)

This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. The goal of this project is to classify images into one of ten categories using CNNs and evaluate the model's performance.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is split into 50,000 training images and 10,000 test images.

### Class Labels:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Libraries Used:
- `NumPy`
- `Matplotlib`
- `TensorFlow`
- `Keras`
- `scikit-learn`
- `seaborn`

## Steps:
1. **Load the Dataset**: The CIFAR-10 dataset is loaded and preprocessed by normalizing the pixel values to a range of 0 to 1.
2. **Data Visualization**: Random sample images from the training dataset are displayed along with their class labels.
3. **Model 1 - Artificial Neural Network (ANN)**: A simple ANN is created for image classification, which includes three dense layers.
4. **Model 2 - Convolutional Neural Network (CNN)**: A CNN is built with two convolutional layers followed by max-pooling layers, flattening, and dense layers for classification.
5. **Training & Evaluation**: Both models are trained, and their performance is evaluated using accuracy and confusion matrix.
6. **Classification Report**: A detailed classification report and confusion matrix are generated to assess the model's performance.

## Results:
The CNN model achieved an accuracy of 69.5% on the test data, showing a significant improvement over the basic ANN model.

### Sample Classification Report:

           precision    recall  f1-score   support
       0       0.46      0.66      0.54      1000
       1       0.61      0.52      0.56      1000
       2       0.36      0.42      0.39      1000
       3       0.41      0.27      0.32      1000
       4       0.41      0.45      0.43      1000
       5       0.51      0.27      0.36      1000
       6       0.59      0.44      0.51      1000
       7       0.63      0.45      0.52      1000
       8       0.64      0.59      0.62      1000
       9       0.39      0.74      0.51      1000
