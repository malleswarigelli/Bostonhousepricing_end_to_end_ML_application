import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# create App
app = Flask(__name__) # starting point of my application from where it will run

# load pickle files i.e model, standard scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('standardscaler.pkl','rb'))

# create app.route
@app.route('/') # first route to home page

def home():
    return render_template('home.html')

# create predict API, is a post request. Collects data using postman, generating o/p after pushing through scalar and regression model)
@app.route('/predict_api', methods=['POST'])

# for API creation
        ## when we give input (json format), capatured into data key. When hit predict_api as a post request, standard scarlar and linear reg kicks in to predict the o/p
def predict_api():
    data = request.json['data']
    print(data)
    print(data.values()) 
    # convert these values into list, then reshape the data as we did in regression.ipynb
    print(np.array(list(data.values())).reshape(1,-1))
    # then apply standard scaler
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    # predict o/p with regression model
    output = regmodel.predict(new_data)
    # since o/p is 2D array, we need 1st value
    print(output[0])
    # return o/p as json
    return jsonify(output[0])

## create small web application, provide inputs, submit the form. As soon as we submit the form, take the data over here and predict the o/p
@app.route('/predict', methods=['POST'])

def predict():
    data = [request.form.values()]
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0] # get 2D array, access 1st value
    return render_template("home.html", prediction_text = "Predicted house price is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)

