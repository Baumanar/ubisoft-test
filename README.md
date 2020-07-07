# Fraud Detection
## Ubisoft home assignment
##### Arnaud Baumann


This project contains different files:


* `assemble_adaboost.ipynb` Notebook with answers to the questions. You need to
run this notebook in order to create a save of the trained assemble.Adaboost model as 
well as a `user_id_counter` dictionnary. These two pickled objects will be reused for the api.

* `utils.py` A python file with helper functions for pre-processing and plots

* `assembleAdaboost.py` A python file containing the Assemble Adaboost model implementation

* `app.py` A python file containing the api

* `Dockerfile` the dockerfile to build the api

* `test_app.py` tests for the api

## API build and run instructions

To build the app in a container, simply run:
`docker build -t fraud_detection .`
Then to run it run: `docker run -d -p 5000:5000 fraud_detection`

You get then make requests on `http://localhost:5000/score`

