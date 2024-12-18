import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt

# Load the model
def load_model(model_path='model.pkl', device=None):
    model = models.resnet50(pretrained=True)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 5)  # Assuming 5 classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Function for prediction
def predict_image(image, model, class_names, device):
    image = preprocess_image(image)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = class_names[preds.item()]
        return predicted_class, probabilities.cpu().numpy()

# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])
    
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'model.pkl'  # Replace with your model file path
    class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']  # Replace with your class names

    if app_mode == "Home":
        st.title("Fruit Image Classifier")
        st.header("Welcome to the Fruit Classifier System!")
        st.image("Fruits_banner.jpg", use_container_width=True)  # Updated parameter

    elif app_mode == "About Project":
        st.title("About the Project")
        st.subheader("Dataset Information")
        st.text("This dataset contains images of various fruits :")
        st.code("Fruits: Apple, Banana, Grape, Mango, Strawberry.")
        st.text("Dataset organized into folders: Train, Test, and Validation.")
        st.text("Each class contains 1,940 training images, 40 validation, and 20 test images.")

    elif app_mode == "Prediction":
        st.title("Model Prediction")
        # Load the model
        model = load_model(model_path, device)

        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter
            
            if st.button("Classify Image"):
                # Prediction
                predicted_class, probabilities = predict_image(image, model, class_names, device)
                
                # Debugging outputs
                st.write(f"Probabilities Array: {probabilities}")
                st.write(f"Predicted Class: {predicted_class}")

                # Display results
                st.write(f"Predicted Class: **{predicted_class}**")
                st.write("Prediction Probabilities:")
                for class_name, prob in zip(class_names, probabilities):
                    st.write(f"{class_name}: {prob:.4f}")
                
                # Plot bar chart
                fig, ax = plt.subplots()
                ax.bar(class_names, probabilities, color='skyblue')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)

if __name__ == "__main__":
    main()
