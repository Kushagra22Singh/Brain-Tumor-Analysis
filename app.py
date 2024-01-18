import os
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from werkzeug.utils import secure_filename

# IMPORTING SQL CODE FOR BACKEND

import mysql.connector as c 
import json

con=c.connect(host='localhost',user='root',passwd='argahsuk@hgnis',database='flask_models')
cursor=con.cursor()

# Create flask app
flask_app = Flask(__name__)
model = tf.keras.models.load_model(r"C:\Users\91811\Desktop\ml with web\final_model.h5")

# Configure the upload folder
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

flask_app.config['UPLOAD_FOLDER'] = upload_folder

# Image data generator for preprocessing
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No image selected')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_text='No image selected')

    if file:
        # Generate a unique filename
        unique_filename = secure_filename(file.filename)

        file_path = os.path.join(flask_app.config['UPLOAD_FOLDER'], unique_filename)

        # Save the file only if the uploads directory exists
        file.save(file_path)

        # Check if the file exists
        if not os.path.exists(file_path):
            return render_template('index.html', prediction_text='File does not exist')

        # Load and preprocess the image using the data generator
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)

        # Map predictions to tumor types
        class_mapping = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
        predicted_class = np.argmax(prediction)

        # Get the tumor type based on the predicted class
        result = class_mapping.get(predicted_class, 'Unknown')

        # INSERTING INTO DATABASE

        json_img_array = json.dumps(img_array.tolist())

        query = "INSERT INTO tumor_data (current_datetime, tumor_name) VALUES (CURRENT_TIMESTAMP, %s)"
        values = (result,)


        cursor.execute(query, values)
        con.commit()


        return render_template("index.html", prediction_text="The predicted tumor type is {}".format(result))

if __name__ == "__main__":
    flask_app.run(debug=False)
