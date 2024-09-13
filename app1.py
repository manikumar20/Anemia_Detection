import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_url_path='/static')

def load_model(model_filename):
    return joblib.load(model_filename)

data = pd.read_csv(r"./uploads/uploaded_file.csv")
df = pd.DataFrame(data)

X = df.drop(columns=['All_Class', 'HGB_Anemia_Class', 'Iron_anemia_Class', 'Folate_anemia_class', 'B12_Anemia_class'])

y = df['All_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 39)

scalar = StandardScaler()
X_trained_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

label = LabelEncoder()
y_train_enc = label.fit_transform(y_train)
y_test_enc = label.transform(y_test)

@app.route('/contactUs')
def contactUs():
    return render_template('contactUs.html')

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

    for i in user_input:
        if i != 'GENDER':
            user_input[i] = float(user_input[i])

    predictions_rf, predictions_gb, dnn_predictions = predict_Dlresult(user_input)
    
    combined_predictions = (predictions_rf + predictions_gb + dnn_predictions) / 3.0
    predicted_anemia = np.round(combined_predictions)
    predicted_class = (predictions_rf)
    return render_template('result.html', predicted_anemia=predicted_anemia, predicted_class=predicted_class, predicted_anemia_dl=dnn_predictions, rbc= user_input['RBC'], hgb= user_input['HGB'], hct= user_input['HCT'], mcv= user_input['MCV'], mch= user_input['MCH'], mchc= user_input['MCHC'], rdw= user_input['RDW'], folate= user_input['FOLATE'], b12= user_input['B12'])

def predict_Dlresult(user_input):    
    filename = "./uploads/uploaded_file.csv"
    df = pd.read_csv(filename)
    feature_names = [
        'GENDER', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'FOLATE', 'B12'
    ]
    
    user_data = pd.DataFrame(user_input, index=[0])
    
    X_train = df[feature_names]
    y_train = df['All_Class']
    
    # Modify GENDER feature
    user_data['GENDER'] = 1 if user_input['GENDER'] == 'Male' else 0

    user_data_dnn = {
        'GENDER': [user_input['GENDER']],
        'RBC': [user_input['RBC']],
        'HGB': [user_input['HGB']],
        'HCT': [user_input['HCT']],
        'MCV': [user_input['MCV']],
        'MCH': [user_input['MCH']],
        'MCHC': [user_input['MCHC']],
        'RDW': [user_input['RDW']],
        'FOLATE': [user_input['FOLATE']],
        'B12': [user_input['B12']]
    }

    user_data_dnn['GENDER'] = [1] if user_input['GENDER'] == 'Male' else [0]

    user_data_dnn = pd.DataFrame(user_data_dnn, index=[0])

    testing_scaled = scalar.transform(user_data_dnn)
    
    X_test = user_data[feature_names]
    X_test_dnn = user_data_dnn[feature_names]
    
    rf_model = load_model('./rf_model.joblib')
    gb_model = load_model('./gb_model.joblib')
    dnn_model = tf.keras.models.load_model('./dnn_model3.h5')
    
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    predictions_rf = rf_model.predict(X_test)
    predictions_gb = gb_model.predict(X_test)

    predictions_dnn = dnn_model.predict(testing_scaled)
    predicted_labels = np.argmax(predictions_dnn, axis=1)

    dnn_predictions = np.round(predictions_dnn).astype(int)
    predicted_labels = np.argmax(dnn_predictions, axis=1)
    
    accuracy_rf = accuracy_score(y_test_base, rf_model.predict(X_test_base))
    accuracy_gb = accuracy_score(y_test_base, gb_model.predict(X_test_base))

    return int(predictions_rf[0]), int(predictions_gb[0]), int(predicted_labels[0])

if __name__ == "__main__":
    app.run(debug=True)