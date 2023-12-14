from tensorflow import keras
from keras import backend as K

from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import tensorflow.keras.backend as K
from flask_cors import CORS
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import json
import requests

import numpy as np
import base64
import io 

app = Flask(__name__)
CORS(app)

modeldir = "modelo/intento1/"
model = load_model(modeldir)

image_size = (180, 180)
batch_size = 128

@app.route('/', methods=['GET'])
def hello():
    print("Se accedió a la ruta /")
    return "Hola, mundo!"


@app.route('/predict', methods=['POST'])
def predict():
        a = request.files['image']
        image_bytes = a.read()

        image_bytes = base64.b64decode(image_bytes)

        img = Image.open(io.BytesIO(image_bytes))

        img = img.resize(image_size)

        devolver = predecir(img)
        print(devolver,999)
        #img = keras.utils.load_img(img, target_size=image_size)
  
        #plt.imshow(img)

        #img_array = keras.utils.img_to_array(img)
#
        #img_array = K.expand_dims(img_array, 0)
        #predictions = model.predict(img_array)
        #score = float(K.sigmoid(predictions[0][0]))
#
        #resultado = (f"This image is {100 * (1 - score):.2f}% Carla and {100 * score:.2f}% Christian.")
        #print(resultado)
        #print(f"This image is {100 * (1 - score):.2f}% Carla and {100 * score:.2f}% Christian.")

        return devolver 
    
sigmoid = (
  lambda x:1 / (1 + np.exp(-x)),
  lambda x:x * (1 - x)
  )

SERVER_URL = 'https://catdog-model-service-krystian-morningstar.cloud.okteto.net/v1/models/catdog-model:predict'

score = 0
        
def predecir(imagen):
    img = imagen
    img = img.resize((180,180))
    img_array = asarray(img)

    img = np.expand_dims(img_array, 0).tolist()
    predict_request = json.dumps({'instances': img })

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 1
    index = 0
    for _ in range(num_requests):
      response = requests.post(SERVER_URL, data=predict_request)
      response.raise_for_status()
      total_time += response.elapsed.total_seconds()
      prediction = response.json()['predictions']
      score = float(sigmoid[0](prediction[0][0]))

      print(response.json())
      print ('sigmoid ', sigmoid[0](prediction[0][0]))
      
      result = (f"This image is {100 * (1 - score):.2f}% Carla and {100 * score:.2f}% Christian.")
      return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)