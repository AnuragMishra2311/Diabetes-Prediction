import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    initial_features=[float(i) for i in request.form.values()]
    final_features=np.array(initial_features).reshape(1,-1)
    prediction=model.predict(final_features)
    
    output=prediction[0]
    
    if output==1:
        return render_template('index.html',prediction_text='You have diabetes')

    else:
        return render_template('index.html',prediction_text="You Don't Have Diabetes")
        
if __name__=='__main__':
    app.run(debug=True)
    