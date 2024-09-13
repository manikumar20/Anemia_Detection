import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dnn_model_path = "C:/Users/sanju/OneDrive/Desktop/dnn_model3.h5"
data = pd.read_csv(r"./uploads/uploaded_file.csv")
df = pd.DataFrame(data)


X = df.drop(columns=['All_Class', 'HGB_Anemia_Class',	'Iron_anemia_Class', 'Folate_anemia_class', 'B12_Anemia_class'])

y = df['All_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 39)

scalar = StandardScaler()
X_trained_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

label = LabelEncoder()
y_train_enc = label.fit_transform(y_train)
y_test_enc = label.transform(y_test)

model = tf.keras.models.load_model(dnn_model_path)

testing = pd.DataFrame({
    'GENDER': [1],
    'RBC': [4.6],
    'HGB': [12.1],
    'HCT': [37.6],
    'MCV': [81.7],
    'MCH': [26.3],
    'MCHC': [32.2],
    'RDW': [15.4],
    'FOLATE': [3.51],
    'B12': [217.6],
})

testing_scaled = scalar.transform(testing)

predictions = model.predict(testing_scaled)
predicted_labels = np.argmax(predictions, axis=1)