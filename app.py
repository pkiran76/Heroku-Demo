import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle  #pickle needed here to read the pickle file created
#use either flask or django to deploy a file/to create the webapp(which is nothing but deployment
app = Flask(__name__) #creation of a flask file using FLASK, this is the flask name
model = pickle.load(open('model.pkl', 'rb'))
#rendering of home page thorugh index.html-all the input fields will be displayed
@app.route('/')
def home():
    return render_template('index.html')
#this is rendered to give us the predicted output
#it uses the predict function(/predict) to get the data from the input fields and give out prediction
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #request library is used to fetch the input data fro all the forms
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
#render the index.html to output the text as {{ prediction_text }} in the index.html file
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

#main function to run this complete flask
if __name__ == "__main__":
    app.run(debug=True)


