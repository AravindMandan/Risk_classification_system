from flask import Flask, render_template, request
import numpy as np
import joblib

# Load model
model = joblib.load('best_final_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect and convert input values to float
    input_values = [float(x) for x in request.form.values()]
    print("Received input length:", len(input_values))  # Optional debug

    # Convert to 2D array for prediction
    input_array = np.array([input_values])

    # Predict
    
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    # Result formatting
    result = 'Approved ✅' if prediction == 1 else 'Rejected ❌'

    return render_template('index.html',
                       prediction_text=f'Loan Prediction: {result}',
                       probability=f'Confidence: {probability:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
