# Sports-Image-Classification
This project applies deep learning models—VGG16, ResNet50, and EfficientNetB0—to classify sports images into 100 categories. It leverages transfer learning, data augmentation, and model evaluation techniques to deliver accurate and efficient results.
#Features

- Classification of images into 100 different sports categories.
- Pre-trained models (VGG16, ResNet50, EfficientNetB0) used with transfer learning.
- Data preprocessing and augmentation for better generalization.
- Real-time prediction via a user-friendly Streamlit interface.
- Performance comparison and ensemble modeling.
#Models Used

- **VGG16** – For simple and detailed pattern recognition.
- **ResNet50** – For deeper learning with residual connections.
- **EfficientNetB0** – For optimized accuracy with lightweight architecture.
#Dataset

- Dataset: [Kaggle – Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- 100 categories, nearly 14,000 images.
- Images resized to 224x224, normalized and augmented.

Run app
  streamlit run app/streamlit_app.py
