import unittest
from app import app
import json


class SignupTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    # Test with valid json payload
    def test_successful_classif(self):
        dict_payload = {"order_id": 1203,
                        "user_id": 2978440680,
                        "order_created_datetime": "15/01/2019 02:36",
                        "amount": 44.066017150878906,
                        "total_amount_14days": 0.0,
                        "email_handle_length": 9,
                        "email_handle_dst_char": 6,
                        "total_nb_orders_player": 7,
                        "player_seniority": 287,
                        "total_nb_play_sessions": 45,
                        "geographic_distance_risk": 1
                        }

        payload = json.dumps(dict_payload)
        response = self.app.post('/score/', headers={"Content-Type": "application/json"}, data=payload)
        assert response.status_code == 200
        assert response.json['decision'] in ['LEGIT', 'BLOCK']
        assert 0 <= response.json['fraud_probability'] <= 1

    # Test where some fields are missing in json
    def test_bad_json_payload_1(self):
        dict_payload = {"order_id": 1203,
                        "user_id": 2978440680,
                        "order_created_datetime": "15/01/2019 02:36",
                        "amount": 44.066017150878906,
                        "total_amount_14days": 0.0,
                        "email_handle_length": 9,
                        "email_handle_dst_char": 6,
                        "total_nb_orders_player": 7,
                        "player_seniority": 287,
                        }

        payload = json.dumps(dict_payload)
        response = self.app.post('/score/', headers={"Content-Type": "application/json"}, data=payload)
        assert response.status_code == 400
        assert response.data.decode() == 'JSON ERROR: fields mismatch'

    # Test where some fields have wrong names
    def test_bad_json_payload_2(self):
        dict_payload = {"WRONG_FIELD": 1203,
                        "user_id": 2978440680,
                        "order_created_datetime": "15/01/2019 02:36",
                        "amount": 44.066017150878906,
                        "total_amount_14days": 0.0,
                        "email_handle_length": 9,
                        "email_handle_dst_char": 6,
                        "total_nb_orders_player": 7,
                        "player_seniority": 287,
                        "total_nb_play_sessions": 45,
                        "geographic_distance_risk": 1
                        }

        payload = json.dumps(dict_payload)
        response = self.app.post('/score/', headers={"Content-Type": "application/json"}, data=payload)

        assert response.status_code == 400
        assert response.data.decode() == 'JSON ERROR: fields mismatch'

    # Test where some fields have wrong types
    def test_bad_json_payload_3(self):
        dict_payload = {"order_id": 1203,
                        "user_id": 2978440680,
                        "order_created_datetime": "15/01/2019 02:36",
                        "amount": '102',
                        "total_amount_14days": 0.0,
                        "email_handle_length": 9,
                        "email_handle_dst_char": 6,
                        "total_nb_orders_player": 7,
                        "player_seniority": 287,
                        "total_nb_play_sessions": 45,
                        "geographic_distance_risk": 1
                        }

        payload = json.dumps(dict_payload)
        response = self.app.post('/score/', headers={"Content-Type": "application/json"}, data=payload)

        assert response.status_code == 400
        assert response.data.decode() == 'JSON ERROR: types mismatch'
