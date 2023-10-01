import streamlit as st
import plotly.express as px

def app():
    st.image('./Untitled design (1).gif', use_column_width=True)

    st.subheader("💡 Abstract:")

    inspiration = '''
    Deep learning has had a lot of success with Generative Adversarial Networks (GANs) in recent times which are used to generate high-quality outputs that are comparable to the original inputs. GANs have been widely utilised to create new realistic pictures and to improve existing ones. On the other hand, GANs may be used to fool individuals by generating false data. Fake faces made by GANs, for example, may deceive not only humans but also machine learning classifiers. Synthetic photographs for identification and authentication purposes, for example, can be used maliciously.
    
    Furthermore, advanced picture editing software such as Adobe Photoshop allows for the alteration of complicated input photographs as well as the creation of high-quality new images. These techniques have improved to the point that they can now build realistic and intricate false pictures that are difficult to distinguish from the genuine thing. YouTube has step-by-step directions and tutorials for making these sorts of fictitious graphics. As a result, these technologies have the potential to be utilised for defamation, impersonation, and factual distortion. Furthermore, with social media, fraudulent material may be swiftly and extensively shared on the Internet.
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    SAFIRE is a tool that utilizes the power of Deep Learning to distinguish Real images from the Fake ones. For instance, if someone takes your original image and inserts your face into a murder scene or photoshops it onto someone else's body, SAFIRE will tag it as fake reducing the chances of it being used to smear you. <br>
    Simply submit the image, and the machine learning model will evaluate it and provide a response in a fraction of a second.'''

    st.markdown(what_it_does, unsafe_allow_html=True)

    stats, buff, graph = st.columns([2, 0.5, 2])

    stats.subheader("🧠 ML Process")

    stats.markdown("<h5> 📊 Getting the Data and EDA Process </h5>", unsafe_allow_html=True)

    stats.markdown("*The dataset was taken from Kaggle and you can find it [here](https://www.kaggle.com/datasets/sophatvathana/casia-dataset).*")

    stats.markdown('''The dataset for this project is the CASIA v2 Image tempering detection dataset
    <li> 7492 Authentic images </li> 
    <li> 5125 Tempered images </li>
    ''', unsafe_allow_html=True) 

    fig = px.bar(x=['Real', 'Fake'], y=[7492, 5125], height=400)
    graph.plotly_chart(fig, use_container_width=True)

    st.write('The dataset for this project is the CASIA v2 Image tempering detection dataset which can be found <a href="https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset"> here </a> on kaggle.  ')

    st.image('./download (1).png', use_column_width=True)
    st.image('./download (2).png', use_column_width=True)
    st.subheader("🖼️CREATING ELA IMAGES FROM THE AUTHENTIC AND TEMPERED IMAGES")
    st.image('./download (3).png', use_column_width=True)
    st.write('ELA helps identify tampered regions in an image, while CNN is trained to classify images as real or fake.')
    st.subheader("⚙️ Model Architecture")
    st.subheader("Metadata-Based Classifier: Extraction of metadata from images and build a metadata-based classifier using Random Forest.")
    st.sidebar.title("Metadata-Based Image Classifier")
    metadata = {
        "ImageWidth": [200],
        "ImageLength": [200],
        "DateTimeOriginal": ["2023-09-30 12:00:00"],
        "Make": ["Canon"],
        "Model": ["EOS Rebel T7i"],
        "Software": ["Adobe Photoshop"],
    }

    # Convert the metadata dictionary into a DataFrame
    import pandas as pd
    metadata_df = pd.DataFrame(metadata)

    # Display the metadata table
    st.table(metadata_df)
    st.write("Accuracy of metadata classifier:60.60%")
    st.image('./model.png', use_column_width=True)
    
    ml_process = f'''
- We designed a Sequential Model having 5 Convolutional Layers and 4 Dense Layers.
- The first layer started with 32 filters and kernel of 2x2.
- The number of filters are doubled at every next layer and kernel is is incremented by 1.
- We introduced some Max Pooling Layers after Convolutional Layers to avoid over-fitting and reduce Computational Costs.
- The Output from Covolutional Layer is Flattened and passed over to Dense Layers.
- We started with 512 neurons in the first Dense layer and reduced them to half over the next two Dense layers.
- Some Dropout Layers were also introduced throught the model to randomly ignore some of the neurons and reduce over-fitting.
- We used ReLU activation in all layers except output layer to reduce computation cost and introduce non-linearity.
- Finally the Output Layer was constructed containing 2 neurons (1 for each class) and softmax activation.
'''
    st.write(ml_process)
    st.image('./download.png', use_column_width=True)
    results = f'''
    - The model with least Validation Loss was saved during the training and reloaded before obtaining the final results.
    '''
    st.subheader("📈 Results")
    st.markdown(results, unsafe_allow_html=True)

    st.write("For our applicaton usecase, We would want the application to perform equally great at both precision and recall. We would want it to attempt to identify all the tempered images while still reducing the errors of classifying authentic images as tempered. Hence, accuracy is a great metric for us. From the training we are able to achieve an accuracy of between 96 and 98% on training dataset and between 90 and 92% on test dataset.")
    
    
    st.write(" ")

    st.write("*Try it out now by clicking on Classify Image button on the Sidebar*")
    st.write("""
    ### About this App
    This web application uses a machine learning model to classify images as fake or real. It's part of a project aimed at detecting manipulated or generated images to combat misinformation.

    Feel free to navigate between pages, upload images, and explore the app!
""")
    # Display footer image
    footer_image_path = './images.png'
    st.image(footer_image_path, caption="scit", width=800)

# Call the app function to run Streamlit app
if __name__ == "__main__":
    app()
