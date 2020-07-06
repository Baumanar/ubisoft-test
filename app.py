from flask import Flask, request, jsonify
import pickle as p
from utils import preprocess, optimal_decision
import numpy as np


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


expectedFields = ["user_id", "order_created_datetime", "amount", "total_amount_14days", "email_handle_length", \
                  "email_handle_dst_char", "total_nb_orders_player", "player_seniority", \
                  "total_nb_play_sessions", "geographic_distance_risk"]


def check_json(expected_fields, data):
    """
    Check if sent json has correct fields
    :param expected_fields: expected fields of json
    :param data: sent json
    :return: Error message if fields are incorrect
    """
    if not data:
        return 'JSON ERROR no data provided'
    if len(data) != 11:
        return 'JSON ERROR  number of fields mismatch'
    for f in expected_fields:
        try:
            data[f]
        except:
            return 'JSON ERROR  field not found: {}'.format(f)


app = Flask(__name__)


@app.route('/score/', methods=['POST'])
def make_predict():
    """
     Method handling for 'score' endpoint, receives a json payload containing all the variables from the table
    """

    data = request.json
    # Check if json is valid and return error message if not
    errorMessage = check_json(expectedFields, data)
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
    # Load model and user counter dict
    modelfile = 'assemble_adaboost_model.pkl'
    model = p.load(open(modelfile, 'rb'))
    user_id_counter_file = 'user_id_counter.pkl'
    user_id_counter = p.load(open(user_id_counter_file, 'rb'))
    app.run(debug=True, host='0.0.0.0')
