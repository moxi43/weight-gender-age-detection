import secrets
import flask
import io
import string
import time
import os
import numpy as np
import torch
import cv2

from typing import Dict, Type
from flask import Flask, jsonify, request

from network_weight import UNet
from preproc import define_model, predict_result
from werkzeug.utils import secure_filename


app = Flask(__name__)

app.config['UPLOADED_PHOTOS'] = 'static'
model = define_model('models/model_ep_37.pth.tar')

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOADED_PHOTOS'], filename))


    if not file:
        return

    return jsonify(prediction=predict_result(os.path.join(app.config['UPLOADED_PHOTOS'], file.filename), model))


@app.route('/', methods=['GET'])
def index():
    return 'sup'
  
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)