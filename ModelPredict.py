import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r"./uploads/uploaded_file.csv")
df = pd.DataFrame(data)


X = df.drop(columns=['All_Class', 'HGB_Anemia_Class',	'Iron_anemia_Class', 'Folate_anemia_class', 'B12_Anemia_class'])

y = df['All_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 39)

print("kchjvgxckv",X_train, X_test, y_train, y_test )

scalar = StandardScaler()
X_trained_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

label = LabelEncoder()
y_train_enc = label.fit_transform(y_train)
y_test_enc = label.transform(y_test)

print(y_train,"\ndsregf.jkdsjghdsfl", y_train_enc)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

prd = model.fit(X_trained_scaled, y_train_enc, epochs = 50, batch_size = 16, validation_split = 0.2)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_enc)
print(f"Test ACCuracy: {test_accuracy * 100}")

predictions = model.predict(X_test_scaled)

predicted_labels = np.argmax(predictions, axis = 1)

predicted_classes = label.inverse_transform(predicted_labels)

# testing = pd.DataFrame({
#     'GENDER': [1],
#     'RBC': [5.57],
#     'HGB': [15.87],
#     'HCT': [47.43],
#     'MCV': [85.14],
#     'MCH': [28.48],
#     'MCHC': [33.45],
#     'RDW': [11.57],
#     'FOLATE': [6.43],
#     'B12': [162.5],
# })

# testing = pd.DataFrame({
#     'GENDER': [0],
#     'RBC': [4.16],
#     'HGB': [11.8],
#     'HCT': [35.1],
#     'MCV': [84.4],
#     'MCH': [28.4],
#     'MCHC': [33.6],
#     'RDW': [12.9],
#     'FOLATE': [10.02],
#     'B12': [675.9],
# })


# testing = pd.DataFrame({
#     'GENDER': [0],
#     'RBC': [3.65],
#     'HGB': [8.18],
#     'HCT': [26.25],
#     'MCV': [71.83],
#     'MCH': [22.39],
#     'MCHC': [31.18],
#     'RDW': [18.5],
#     'FOLATE': [8.37],
#     'B12': [157.5],
# })


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

# testing = pd.DataFrame({
#     'GENDER': [1],
#     'RBC': [4.47],
#     'HGB': [12.9],
#     'HCT': [38.5],
#     'MCV': [86.2],
#     'MCH': [29],
#     'MCHC': [33.6],
#     'RDW': [16.6],
#     'FOLATE': [6.2],
#     'B12': [287.2],
# })

# testing = pd.DataFrame({
#     'GENDER': [1],
#     'RBC': [3.1],
#     'HGB': [13.2],
#     'HCT': [38.3],
#     'MCV': [123.5],
#     'MCH': [42.6],
#     'MCHC': [34.5],
#     'RDW': [12.2],
#     'FOLATE': [15.2],
#     'B12': [247.5],
# })

# testing = pd.DataFrame({
#     'GENDER': 1,
#     'RBC': 3.1,
#     'HGB': 13.2,
#     'HCT': 38.3,
#     'MCV': 123.5,
#     'MCH': 42.6,
#     'MCHC': 34.5,
#     'RDW': 12.2,
#     'FOLATE': 15.2,
#     'B12': 247.5,
# })




# Scale the input data using the same scalar used for training data
testing_scaled = scalar.transform(testing)

# Make predictions on the scaled input data
predictions = model.predict(testing_scaled)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
predicted_classes = label.inverse_transform(predicted_labels)

print(predictions,predicted_labels, predicted_classes)


predicted_anemia = 0
if ( predicted_classes[0] == 0) : 
    predicted_anemia = 0    
else: 
    predicted_anemia = 1
    
print(predicted_anemia)

