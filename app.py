from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import traceback
import os

app = Flask(__name__)
# Enable CORS so your local or deployed index.html can communicate with this backend safely
CORS(app)  

# Load pre-trained artifacts
try:
    with open('salary_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("Successfully loaded model, scaler, and feature definitions.")
except Exception as e:
    print("Error loading pickle files. Make sure salary_model.pkl, scaler.pkl, and features.pkl are in the same directory.")
    print(str(e))

def preprocess_inputs(data, expected_features):
    """
    Robust feature vector builder.
    Maps inputs from the HTML form to the exact columns expected by your Random Forest model,
    including handling numeric defaults and categorical variables.
    """
    row_dict = {col: 0.0 for col in expected_features}
    
    # 1. Parse numeric inputs
    age = float(data.get('age', 30))
    hours = float(data.get('hours', 40))
    
    # Identify numerical columns and assign values
    for col in expected_features:
        col_lower = col.lower()
        if 'age' in col_lower and 'group' not in col_lower:
            row_dict[col] = age
        elif 'hour' in col_lower:
            row_dict[col] = hours
        elif 'capital' in col_lower and 'gain' in col_lower:
            row_dict[col] = 0.0  # Median default capital gain
        elif 'capital' in col_lower and 'loss' in col_lower:
            row_dict[col] = 0.0  # Median default capital loss
        elif 'education' in col_lower and 'num' in col_lower:
            # Map selected education level to numeric standard values
            edu_map = {'HS-grad': 9, 'Assoc-voc': 11, 'Bachelors': 13, 'Masters': 14, 'Doctorate': 16}
            user_edu = data.get('education', 'Bachelors')
            row_dict[col] = float(edu_map.get(user_edu, 10))

    # 2. Parse categorical features (One-Hot Encoded variables)
    selected_education = str(data.get('education', 'Bachelors'))
    selected_occupation = str(data.get('occupation', 'Tech-support'))
    
    for col in expected_features:
        if '_' in col:
            parts = col.split('_')
            category_prefix = parts[0].lower()
            category_value = '_'.join(parts[1:]).lower()
            
            if 'education' in category_prefix and category_value == selected_education.lower():
                row_dict[col] = 1.0
            elif 'occupation' in category_prefix and category_value == selected_occupation.lower():
                row_dict[col] = 1.0
        elif col.lower() == 'education':
            edu_label_map = {'HS-grad': 1, 'Assoc-voc': 2, 'Bachelors': 3, 'Masters': 4, 'Doctorate': 5}
            row_dict[col] = float(edu_label_map.get(selected_education, 1))
        elif col.lower() == 'occupation':
            occ_label_map = {'Tech-support': 1, 'Sales': 2, 'Craft-repair': 3, 'Prof-specialty': 4, 'Exec-managerial': 5}
            row_dict[col] = float(occ_label_map.get(selected_occupation, 1))

    # Convert sorted row values to a 2D array matching the precise column order of training features
    feature_vector = [row_dict[col] for col in expected_features]
    return np.array([feature_vector])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input variables provided'}), 400

        # Construct and align the incoming data with expected features
        raw_features = preprocess_inputs(data, feature_columns)
        
        # Apply the StandardScaler exactly like the training pipeline
        scaled_features = scaler.transform(raw_features)
        
        # Execute prediction
        prediction = model.predict(scaled_features)
        
        # Translate predictions to strings (supporting continuous or binary targets)
        result = ">50K" if int(prediction[0]) == 1 else "<=50K"
        
        return jsonify({
            'prediction': result,
            'status': 'success'
        })
        
    except Exception as e:
        print("Error during inference:")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

@app.route('/', methods=['GET'])
def index():
    return "Salary Prediction API is active and running!"

if __name__ == '__main__':
    # Dynamically bind to host port (critical for platforms like Render/Heroku)
    port = int(os.environ.get('PORT', 5000))
    # We disable the reloader (use_reloader=False) to prevent threading/signal issues in restricted cloud environments
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
