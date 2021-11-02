import pickle

import numpy as np

from flask import Flask, request, jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('boardgame_ratings')

@app.route('/predict', methods=['POST'])
def predict():
    boardgame = request.get_json()

    X = dv.transform([boardgame])
    y_pred = model.predict(X)

    result = {
        'predicted_rating': y_pred.tolist()[0]
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)