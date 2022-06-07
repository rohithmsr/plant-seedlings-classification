import os

from dotenv import load_dotenv
load_dotenv()

# Flask
from flask import Flask, request, render_template, jsonify

# TensorFlow and tf.keras
from tensorflow.keras.models import load_model

# Some utilites
import numpy as np
import boto3
import cv2
from util import base64_to_pil

# env vars
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Declare a flask app
app = Flask(__name__)

from tensorflow.keras.models import load_model
MODEL_NAME = 'plantspecies_CNN_model.h5'
MODEL_PATH = 'models/my_image_classifier_from_s3.h5'

SPECIES = [
    'Black-grass', 
    'Charlock', 
    'Cleavers', 
    'Common Chickweed', 
    'Common wheat', 
    'Fat Hen', 
    'Loose Silky-bent', 
    'Maize', 
    'Scentless Mayweed', 
    'Shepherds Purse', 
    'Small-flowered Cranesbill', 
    'Sugar beet'
]

PLANTS_TYPE = {
    'Scentless Mayweed': 1, 
    'Common wheat': 0, 
    'Charlock': 0, 
    'Black-grass': 1,
    'Sugar beet': 0, 
    'Loose Silky-bent': 0, 
    'Maize': 0, 
    'Cleavers': 1,
    'Common Chickweed': 1, 
    'Fat Hen': 1, 
    'Small-flowered Cranesbill': 1,
    'Shepherds Purse': 1
}

s3 = boto3.resource(
    service_name = 's3',
    region_name='ap-south-1',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# s3.Bucket("image-cloud-asmt").download_file(
#     MODEL_NAME, MODEL_PATH
# )

# # Load your own trained model from the S3 bucket
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img, model):
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img_arr = [cv2.resize(opencvImage, (256, 256))]
    img_X = np.asarray(img_arr)

    # Normalization of the Image Data
    img_X = img_X.astype('float32') / 255

    preds = model.predict(img_X)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)

        pred_class = np.argmax(preds, axis=1)
        # print(pred_class)

        species_name = SPECIES[pred_class[0]]
        species_type = 'WEED' if PLANTS_TYPE[species_name] else 'PLANT'

        result = species_type + ' -> ' + species_name

        # Serialize the result
        return jsonify(
            result=result,
            species_name=species_name, 
            species_type=species_type
        )

    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

