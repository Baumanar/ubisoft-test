from flask import Flask, request, jsonify
import pickle as p
from utils import preprocess, optimal_decision
import numpy as np
import numbers


def data_to_array(data):
    """
    Transforms json input to numpy array
    :param data: json input
    :return: numpy array of input json
    """
    arr = []
    for key in data:
        arr.append(data[key])
    return np.array([arr[1:]])


expectedFields = ["order_id", "user_id", "order_created_datetime", "amount", "total_amount_14days",
                  "email_handle_length", "email_handle_dst_char", "total_nb_orders_player",
                  "player_seniority", "total_nb_play_sessions", "geographic_distance_risk"]
expectedTypes = [numbers.Number, numbers.Number, str, numbers.Number, numbers.Number,
                 numbers.Number, numbers.Number, numbers.Number,
                 numbers.Number, numbers.Number, numbers.Number]


def check_json(expected_fields, expected_types, data):
    """
    Check if sent json has correct fields and types
    :param expected_fields: expected fields of json
    :param expected_types: expected type values in json
    :param data: sent json
    :return: Error message if fields are incorrect
    """
    if sorted(data.keys()) != sorted(expected_fields):
        return 'JSON ERROR: fields mismatch'
    if not all([isinstance(k, expected_types[i]) for i, k in enumerate(data.values())]):
        return 'JSON ERROR: types mismatch'


# Load model and user counter dict
modelfile = 'assemble_adaboost_model.pkl'
model = p.load(open(modelfile, 'rb'))
user_id_counter_file = 'user_id_counter.pkl'
user_id_counter = p.load(open(user_id_counter_file, 'rb'))
app = Flask(__name__)


@app.route('/score/', methods=['POST'])
def make_predict():
    """
     Method handling for 'score' endpoint, receives a json payload containing all the variables from the table
    """

    data = request.json
    # Check if json is valid and return error message if not
    errorMessage = check_json(expectedFields, expectedTypes, data)
    if errorMessage is not None:
        return errorMessage, 400

    # Convert json to array and process user_id and order_created_datetime
    amount = data['amount']
    data = data_to_array(data)
    preprocess(data, user_id_counter)

    # Make prediction and compute decision
    prediction = model.predict_proba(data.astype(float))
    decision = optimal_decision(amount=amount, fraud_fee=15, p=prediction[0][0])

    return jsonify(fraud_probability=prediction[0][0], decision=decision)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
