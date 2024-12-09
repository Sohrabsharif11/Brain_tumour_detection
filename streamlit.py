import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('BrainTumor10Epochs.h5')  # model path 
    return model

model = load_model()

# preprocessing  the image

def preprocess_image(image):
    img = image.resize((64, 64))  # Resize the image to (64, 64)
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Streamlit app
def main():
    st.title('Brain Tumor Detection')

    uploaded_file = st.file_uploader("test\pred0.jpg", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True,width=200)
        st.write("")

        # Make prediction
        if st.button('Diagnose'):
            processed_img = preprocess_image(image)
            result=model.predict(processed_img)
            result_final=np.argmax(result,axis=-1)
            print(result) 
            if result[0][0] >= 0.5:
                st.write('Brain Tumour is  Present ')
            else:
                st.write('Brain Tumour is not Present')

if __name__ == '__main__':
    main()
