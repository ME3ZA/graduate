import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

model_path='r'C:\Users\user\Desktop\Graduation\chest_model_balanced.h5''

st.title("Chest Disease Identification Using CT Scan")
upload = st.file_uploader('Upload a CT scan image')


if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  if(st.button('Predict')):
    model = tf.keras.models.load_model(model_path)
    x = cv2.resize(opencv_image,(100,100))
    x = np.expand_dims(x,axis=0)    
    y = model.predict(x)
    ans=np.argmax(y,axis=1)
    if(ans==0):
      st.title('COVID')
    elif(ans==1):
      st.title('Healthy')
    elif(ans==2):
      st.title('Lung Tumor')
    else:
      st.title('Common Pneumonia')
