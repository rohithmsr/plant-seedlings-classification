import os

from dotenv import load_dotenv
load_dotenv()

# Flask
from flask import Flask, request, render_template, jsonify

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
import boto3
from util import base64_to_pil

# env vars
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Declare a flask app
app = Flask(__name__)

from tensorflow.keras.models import load_model
MODEL_NAME = 'plantspecies_CNN_model.h5'
MODEL_PATH = 'models/my_image_classifier_from_s3.h5'

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
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
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

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

