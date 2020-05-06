from flask import Flask, request
import torch
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

net = torch.jit.load('my_model.zip')


@app.route('/')
def hello():
    return "Hello World!"


@app.route("/predict", methods=['POST'])
def predict():

    # load image
    img = Image.open(request.files['file'].stream).convert(
        'RGB').resize((224, 224))
    img = np.array(img)
    img = torch.FloatTensor(img.transpose((2, 0, 1)) / 255)

    # get predictions
    pred = net(img.unsqueeze(0)).squeeze()
    pred_probas = torch.softmax(pred, axis=0)

    return {
        'malignant': pred_probas[1].item(),
        'bening': pred_probas[0].item()
    }


if __name__ == "__main__":
    app.run(debug=True)
