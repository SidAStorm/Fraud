import os
import numpy as np
from PIL import Image
import streamlit as st
from datetime import datetime
import exifread
from tensorflow import keras

# Streamlit UI
def app():
    st.image('./Untitled design (1).gif', use_column_width=True)
    st.write("Upload a Picture to see if it is a fake or real face.")
    st.markdown('*Need an image to test? Visit this [link]("https://www.kaggle.com/datasets/awsaf49/artifact-dataset")*')
    st.title('Image Fraud Detection')

    # Function to calculate accuracy
    def calculate_accuracy(Y_true, Y_pred):
        threshold = 0.5
        Y_pred_classes = (Y_pred[:, 1] > threshold).astype(int)
        accuracy = np.sum(Y_true == Y_pred_classes) / len(Y_true)
        return accuracy

    # Function to classify an image using the trained model
    def classify_image(image_path, model, threshold=0.5):
        input_shape = (128, 128)  # Adjust the dimensions as per your model's input shape
        image = Image.open(file_path) # reading the image
        image = image.resize(image, input_shape)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model.predict(image)
        predicted_probability = prediction[0][1]  # Probability of being a "Real Image"

        if predicted_probability > threshold:
            return "Real Image"
        else:
            return "Fake Image"

    # Function to perform metadata analysis on an image
    def perform_metadata_analysis(image_path):
        def extract_metadata(image_path):
            with open(image_path, 'rb') as image_file:
                tags = exifread.process_file(image_file)
            return tags

        def extract_metadata_details(image_path):
            metadata = extract_metadata(image_path)
            creation_time = metadata.get('EXIF DateTimeOriginal', None)
            software_used = metadata.get('Image Software', None)
            date_modified_timestamp = os.path.getmtime(image_path)
            date_modified = datetime.fromtimestamp(date_modified_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            return {
                'Creation Date/Time': str(creation_time),
                'Software Used': str(software_used),
                'Date Modified': date_modified,
            }

        metadata_details = extract_metadata_details(image_path)
        return metadata_details

    # Display the uploaded image
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'], key="file_upload")

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Load the pre-trained model
        model_path = './model.h5'
        model = keras.models.load_model(model_path)

        # Classify the uploaded image
        classify_button = st.button('Classify Image')
        if classify_button:
            image_path = 'uploaded_image.jpg'
            with open(image_path, 'wb') as f:
                f.write(uploaded_image.read())

            classification_result = classify_image(image_path, model)

            if classification_result == 'Real Image':
                st.success(f'Image classification result: {classification_result}')
            else:
                st.error(f'Image classification result: {classification_result}')
                
                # Perform metadata analysis
                metadata_details = perform_metadata_analysis(image_path)
                st.subheader('Metadata Analysis:')
                for key, value in metadata_details.items():
                    st.write(f"{key}: {value}")

    # Display footer image
    footer_image_path = './images.png'
    st.image(footer_image_path, caption="scit", width=800)

# Run the Streamlit app
if __name__ == '__main__':
    app()
