# Sports-Image-Classification
This project applies deep learning models—VGG16, ResNet50, and EfficientNetB0—to classify sports images into 100 categories. It leverages transfer learning, data augmentation, and model evaluation techniques to deliver accurate and efficient results.

# Dataset

- Dataset: [Kaggle – Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- 100 categories, nearly 14,000 images.
- Images resized to 224x224, normalized and augmented.
# Features

- Classification of images into 100 different sports categories.
- Pre-trained models (VGG16, ResNet50, EfficientNetB0) used with transfer learning.
- Data preprocessing and augmentation for better generalization.
- Real-time prediction via a user-friendly Streamlit interface.
- Performance comparison and ensemble modeling.

## 🛠️ Technology Stack

- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow, Keras  
- **Pre-trained Models Used**:  
  - VGG16  
  - ResNet50  
  - EfficientNetB0 (via ImageNet)  
- **Web Application Framework**: Streamlit  
- **Data Processing**: NumPy, Pandas  
- **Image Preprocessing & Augmentation**: Keras `ImageDataGenerator`  
- **Model Evaluation**: Scikit-learn (accuracy, precision, recall, F1-score, confusion matrix)  
- **Data Visualization**: Matplotlib, Seaborn  

## 📌 Step-by-Step Procedure

1. **Problem Definition**  
   - Define the goal: Classify images into 100 sports categories using deep learning.

2. **Data Collection**  
   - Download the dataset from [Kaggle – Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification).  
   - Organize images into training, validation, and test folders.

3. **Data Preprocessing**  
   - Resize all images to 224x224 pixels.  
   - Normalize pixel values to the range [0, 1].  
   - Apply data augmentation (rotation, flipping, zooming, brightness adjustment) using `ImageDataGenerator`.  
   - Split the dataset into training, validation, and test sets.

4. **Model Building**  
   - Load pre-trained models (VGG16, ResNet50, EfficientNetB0) with weights from ImageNet.  
   - Add custom classification layers (Dense + Softmax) to output 100 sports categories.

5. **Model Compilation**  
   - Use `categorical_crossentropy` as the loss function.  
   - Use the `Adam` optimizer and monitor `accuracy`.

6. **Model Training**  
   - Train the models on the training data.  
   - Validate performance using the validation set.  
   - Monitor training and validation accuracy/loss over epochs.

7. **Model Evaluation**  
   - Evaluate the model on the test set.  
   - Calculate accuracy, precision, recall, F1-score.  
   - Plot the confusion matrix for detailed insights.

8. **Model Saving**  
   - Save the best-performing model:  
     ```python
     model.save("best_sports_model.h5")
     ```

9. **Web App Deployment**  
   - Create a Streamlit web app for real-time image upload and classification.  
   - Display the predicted sports category and confidence score on the UI.

## Results

1. HomePage of the web application.Click [here](https://drive.google.com/file/d/1a2EyETNrlzH3B0s2XdvCZscE_BIUwStT/view?usp=sharing)
2. After an image is uploaded.Click [here](https://drive.google.com/file/d/1yF0edkAmi67RqeFZ3f6Gy0XJcTN7Czn-/view?usp=sharing)


## Run app

Open anaconda prompt run
  -streamlit run app/streamlit_app.py
