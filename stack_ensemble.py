import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from io import StringIO

app = Flask(__name__, static_url_path='/static')

def train_and_predict(model_class, X_train, y_train, X_test):
    model = model_class()
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    classes = model.classes_

    if 'All_Class' not in classes:
        classes = np.append(classes, 'All_Class')
        predictions = np.hstack((predictions, np.zeros((predictions.shape[0], 1))))

    return pd.DataFrame(data=predictions, columns=classes)

def print_predictions(predictions):
    """Prints the predictions."""
    max_index = predictions.iloc[:, :-1].values.argmax(axis=1)
    all_class_values = max_index
    predictions_with_all_class = predictions.copy()
    predictions_with_all_class['All_Class'] = all_class_values
    return predictions_with_all_class.to_html()

@app.route('/')
def index():
    features = [
        'GENDER', 'RBC', 'HGB', 'HCT',
        'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]
    return render_template('index.html', features=features)

def stacking_ensemble(base_model1, base_model2, meta_model, X_train, y_train, X_test):
    # Split the training data into two sets: base learners and meta learner
    X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    
    # Train the first base model
    base_model1.fit(X_base_train, y_base_train)
    
    # Generate predictions from the first base model
    base_model1_predictions = base_model1.predict_proba(X_meta_train)
    
    # Train the second base model
    base_model2.fit(X_base_train, y_base_train)
    
    # Generate predictions from the second base model
    base_model2_predictions = base_model2.predict_proba(X_meta_train)
    
    # Concatenate the predictions from both base models
    meta_features = np.column_stack((base_model1_predictions, base_model2_predictions))
    
    # Train the meta model on the meta features
    meta_model.fit(meta_features, y_meta_train)
    
    # Generate predictions from the base models on the test set
    base_model1_test_predictions = base_model1.predict_proba(X_test)
    base_model2_test_predictions = base_model2.predict_proba(X_test)
    
    # Concatenate the test set predictions
    test_meta_features = np.column_stack((base_model1_test_predictions, base_model2_test_predictions))
    
    # Generate final predictions using the meta model
    final_predictions = meta_model.predict_proba(test_meta_features)
    
    # Create DataFrame
    classes = base_model1.classes_
    if 'All_Class' not in classes:
        classes = np.append(classes, 'All_Class')
        final_predictions = np.hstack((final_predictions, np.zeros((final_predictions.shape[0], 1))))
    return pd.DataFrame(data=final_predictions, columns=classes)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contactUs')
def contactUs():
    return render_template('contactUs.html')

@app.route('/predict', methods=['POST'])
def predict():
    filename = "./uploads/uploaded_file.csv"
    df = pd.read_csv(filename)
    feature_names = [
        'GENDER', 'RBC', 'HGB', 'HCT',
        'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]
    
    user_input = {feature: request.form.get(feature) for feature in feature_names}
    user_data = pd.DataFrame(user_input, index=[0])

    X_train = df[feature_names]
    y_train = df['All_Class']
    
    rf_model = RandomForestClassifier()
    gb_model = GradientBoostingClassifier()
    meta_model = RandomForestClassifier() 
    
    stacked_predictions = stacking_ensemble(rf_model, gb_model, meta_model, X_train, y_train, user_data)

    all_class_value = str(stacked_predictions.iloc[:, :-1].values.argmax(axis=1)[0])
    
    return render_template('result.html', predicted_class=all_class_value)

if __name__ == "__main__":
    app.run(debug=True)