from flask import Flask,request,app,jsonify,url_for,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
model = pickle.load(open('model.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictAPI',methods=['POST'])
def predictAPI():
    data =request.json['data']
    print(data)
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("index.html",prediction_text="Then price of the house is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)
