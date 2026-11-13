import pickle

import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


features = list(dv.get_feature_names_out())

app = Flask('pitnextlap')

@app.route('/predict', methods=['POST'])
def predict():
    observation = request.get_json()

    observation = {
        key.lower().replace(' ', '_'):
        value.lower().replace(' ', '_') if isinstance(value, str) else value
        for key, value in observation.items()
    }

    X = dv.transform([observation])
    dX = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dX)[0]

    pitnextlap = y_pred >= 0.5

    result = {
        'pit_probability': float(y_pred),
        'pitnextlap': bool(pitnextlap)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)