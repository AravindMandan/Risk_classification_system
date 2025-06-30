from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib

# Load model and scaler
model = joblib.load('best_final_model.pkl')


#model, scaler = pickle.load(open('best_final_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs from form
    input_values = [float(x) for x in request.form.values()]
    
    # Scale the inputs using saved scaler
    final_input = scaler.transform([input_values])
    
    # Predict
    prediction = model.predict(final_input)[0]
    
    result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'
    
    return render_template('index.html', prediction_text=f'Loan Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
