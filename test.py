import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from io import StringIO
import joblib

app = Flask(__name__, static_url_path='/static')

def load_model(model_filename):
    return joblib.load(model_filename, mmap_mode='r')


def predict_input(user_input, rf_model, gb_model, dnn_model):
    # Create input data as DataFrame
    user_data = pd.DataFrame(user_input, index=[0])
    
    # Modify GENDER feature
    user_data['GENDER'] = 1 if user_input['GENDER'] == 'Male' else 0
    
    # Extract feature names
    feature_names = [
        'GENDER', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]
    
    # Prepare input data
    X_input = user_data[feature_names]
    
    # Predict using each model
    prediction_rf = rf_model.predict(X_input)
    prediction_gb = gb_model.predict(X_input)
    prediction_dnn = dnn_model.predict(X_input)
    
    # Print predictions
    print("Random Forest Prediction:", prediction_rf)
    print("Gradient Boosting Prediction:", prediction_gb)
    print("DNN Prediction:", prediction_dnn)

    # Combine predictions
    # Assuming all models predict probabilities for binary classification
    combined_predictions = (prediction_rf + prediction_gb + prediction_dnn) / 3.0
    
    # Get the class with the highest probability
    predicted_class = np.round(combined_predictions)[0][0]
    
    return predicted_class

@app.route('/')
def index():
    features = [
        'GENDER', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])

def predict():
    filename = "./uploads/uploaded_file.csv"
    df = pd.read_csv(filename)
    feature_names = [
        'GENDER', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]

    user_input = {feature: request.form.get(feature) for feature in feature_names}
    user_data = pd.DataFrame(user_input, index=[0])
    print("\n\n", user_data)
    
    X_train = df[feature_names]
    y_train = df['All_Class']
    
    # Modify GENDER feature
    user_data['GENDER'] = 1 if user_input['GENDER'] == 'Male' else 0
    
    X_test = user_data[feature_names].astype(float)  # Ensure data type is float
    
    rf_model = RandomForestClassifier()
    gb_model = GradientBoostingClassifier()
    dnn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Placeholder model for demonstration
    placeholder_model = RandomForestClassifier()  
    
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    rf_model.fit(X_train_base, y_train_base)
    gb_model.fit(X_train_base, y_train_base)
    dnn_model.fit(X_train_base, y_train_base, epochs=50, batch_size=32, validation_split=0.2)
    
    predictions_rf = rf_model.predict(X_test)
    predictions_gb = gb_model.predict(X_test)
    predictions_dnn = dnn_model.predict(X_test)
    
    print("Random Forest Predictions:", predictions_rf)
    print("Gradient Boosting Predictions:", predictions_gb)
    print("DNN Predictions:", predictions_dnn)
    
    combined_predictions = (predictions_rf + predictions_gb + predictions_dnn) / 3.0
    predicted_class = np.round(int(combined_predictions[0][0]))
    
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
