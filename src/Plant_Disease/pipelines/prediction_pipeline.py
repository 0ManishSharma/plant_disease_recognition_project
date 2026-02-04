import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

def predict_image(image_path):
    model = tf.keras.models.load_model("artifacts/model.h5")

    with open("artifacts/class_names.json",'r') as f:
        class_name = json.load(f)
    img = image.load_img(image_path,target_size=(224,224))
    img_arr = image.img_to_array(img)
    img_arr = img_arr/255.0
    img_arr = np.expand_dims(img_arr,axis=0)

    # Predict
    prediction = model.predict(img_arr)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return class_name[predicted_index],round(confidence,2)