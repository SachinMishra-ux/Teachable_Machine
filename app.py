from keras.models import load_model
import streamlit as st
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

"""
# deep Classifier project
"""

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_Model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#image = Image.open('MC461.jpeg').convert('RGB')
uploaded_file = st.file_uploader("Choose a file")
#image = Image.open(uploaded_file)#.convert('RGB')
#image=image.convert('RGB')
if uploaded_file is not None:
    image = Image.open(uploaded_file)

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    #imgage = image.resize((224,224))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    st.image(image, caption=class_name)
    st.image(image,caption=confidence_score)
    print('Class:', class_name, end='')
    print('Confidence score:', confidence_score)