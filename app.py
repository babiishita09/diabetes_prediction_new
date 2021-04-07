



#import the necessary dependencies

from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np

app = Flask(__name__) #initializing a flask app
@app.route('/')   #route to display the homepage
@cross_origin()
def homepage():
    return render_template('index.html')

@app.route("/predict", methods = ["POST","GET"])    #route to show the prediction in a web ui
@cross_origin()
def predict():
    if request.method == "POST":
        # reading the input given by the user
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])
        filename = 'modelForPrediction.sav'
        loaded_model = pickle.load(open(filename,'rb')) #loading the model file from the storage
        #prediction using the loaded model file
        data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        my_prediction = loaded_model.predict(data)
        if my_prediction[0] == 1:
            prediction = 'positive'
        else:
            prediction = 'negative'

        return render_template('result.html',prediction=prediction)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(debug=True)



