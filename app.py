import streamlit as st
import tensorflow as tf
import streamlit as st
import keras
from PIL import Image, ImageOps
import numpy as np

modelpath = 'C:/Users/px/Desktop/Covid Project/models/covidtestmodel.h5'
model = keras.models.load_model(modelpath)
class_names = ['Emphysema and bronchiectasis',
 'ILD',
 'Old TB',
 'active pulmonary tuberculosis',
 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
 'normal',
 'old  pulmonary tuberculosis, with post infective bronchiectasis',
 'old  pulmonary tuberculosis, with post tubercular bronchiectasis',
 'pneumonia and bronchiectasis',
 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']

st.write("Covid Classifier")
file = st.file_uploader("Please upload a lung scan file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_np = np.array(image)
    dimension = len(image_np.shape)
    if dimension == 2:
        converted_image = np.expand_dims(image_np, axis=2)
    elif dimension == 4:
        converted_image = image_np[:, :, :3]
    else:
        converted_image = image_np

    resize = tf.image.resize(converted_image, (256, 256))
    
    if resize.shape[-1] == 4:
        resize = resize[:, :, :3]

    # Convert the image to RGB format before resizing
    if resize.shape[-1] == 1:
        resize = tf.image.grayscale_to_rgb(resize)

    resized_image = np.expand_dims(resize, 0)
    value = model.predict(resized_image / 255.0)
    val = value.tolist()
    val = max(val)
    predicted_index = (val.index(max(val)))
    st.write(max(val))
    predicted_class = class_names[predicted_index]
    predicted_class
    st.write(predicted_class)
