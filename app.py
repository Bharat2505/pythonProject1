#packages
import numpy as np
from flask import Flask,render_template, request
import pickle

app= Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@app.route('/')

#functions
def home():
    return render_template('index.html')
@app.route('/predict', methods=['post'])

def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html',
                           prediction_text='fuel price for kilometer driven is:{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
