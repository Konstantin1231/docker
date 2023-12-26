from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np
import pickle
from joblib import load

# always start with:
app_1 = Flask(__name__)  # applyies when meet __name__

classifier = load("clf.joblib")


@app_1.route("/")  # if methode not provided: get by default
def home():
    return render_template('home.html')


@app_1.route("/predict_api", methods=['POST'])
def predict_api():
    # four features to provide
    data = request.json["data"]
    print(data)
    prediction = classifier.predict(np.array(list(data.values)))
    return jsonify(prediction)

@app_1.route("/predict", methods=['POST'])
def predict():
    # four features to provide
    data = [float(x) for x in request.form.values()]
    data = np.array(data).reshape(1,-1)
    prediction = classifier.predict(data)
    return render_template("home.html", prediction_text="Prediction is {}".format(prediction))



if __name__ == "__main__":
    app_1.run()
