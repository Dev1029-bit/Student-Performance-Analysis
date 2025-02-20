from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the Model
model = joblib.load("model/student_performance_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect Data from Form
    features = [
        float(request.form['math_score']),
        float(request.form['reading_score']),
        float(request.form['writing_score']),
        float(request.form['lunch']),
        float(request.form['test_preparation'])
    ]
    final_features = [np.array(features)]
    
    # Make Prediction
    prediction = model.predict(final_features)
    
    # Output Result
    output = "Pass" if prediction[0] == 1 else "Fail"
    return render_template('result.html', prediction_text=f'Student Performance: {output}')

if __name__ == '__main__':
    app.run(debug=True)
