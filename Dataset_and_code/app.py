from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("waste_model.h5")
labels = ['biodegradable', 'recyclable', 'trash']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    filepath = os.path.join("uploads", f.filename)
    os.makedirs("uploads", exist_ok=True)
    f.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = labels[np.argmax(prediction)]

    return f"<h2>Prediction: {result}</h2><br><a href='/'>Try Another</a>"

if __name__ == '__main__':
    app.run(debug=True)
