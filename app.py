from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Inisiasi data
data = pd.read_csv('data.csv')

df = pd.DataFrame(data)

# Penambahan data dummy
# dummy_data = {
#     'age': [22, 24, 25, 28, 29, 31, 34, 37, 41, 42,
#             15, 20, 20, 30, 35, 35, 45, 50, 55, 60,
#             ],
#     'gadgetPerHour': [8, 9, 6, 5, 7, 7, 5, 4, 3, 4,
#                       4, 8, 8, 4, 6, 6, 4, 4, 2, 2,
#                       ],
#     # wrongLens: 0 = Tidak; 1 = Ya
#     'wrongLens': [0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
#                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                   ],
#     # genFactor: 1 = Tidak; 2 = Kurang Tahu; 3 = Ya
#     'genFactor': [1, 3, 2, 1, 3, 3, 1, 2, 3, 1,
#                   1, 1, 1, 3, 1, 2, 1, 1, 1, 1,
#                   ],
#     # nutriFood: 1 = Tidak; 2 = Jarang; 3 = Ya
#     'nutriFood': [3, 1, 2, 2, 1, 2, 2, 3, 2, 3,
#                   3, 2, 3, 2, 3, 2, 3, 2, 3, 2,
#                   ],
#     'eyeDisease': [0, 1, 1, 0, 1, 1, 0, 0, 1, 0,
#                    1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
#                    ],
# }

# df_dummy = pd.DataFrame(dummy_data)

# df = pd.concat([df, df_dummy], ignore_index=True)

# Penanganan missing value dari data dummy
df = df.fillna(0)

# Pengkategorian nilai data
df['sex'] = df['sex'].replace({'Laki-laki' : 0, "Perempuan" : 1})
df['eyeDisease'] = df['eyeDisease'].replace({'Tidak' : 0, "Ya" : 1})
df['typeOfEyeDisease'] = df['typeOfEyeDisease'].replace({"Rabun Jauh (Mata Minus)" : 1, "Rabun Dekat (Mata Plus)" : 2, "Mata Silinder" : 3, "Rabun Jauh (Mata Minus), Mata Silinder" : 4, "Rabun Dekat (Mata Plus), Mata Silinder" : 5, "Rabun Jauh (Mata Minus), Rabun Dekat (Mata Plus), Mata Silinder" : 6,})
df['wrongLens'] = df['wrongLens'].replace({'Tidak' : 0, "Ya" : 1})
df['darkAct'] = df['darkAct'].replace({'Tidak' : 0, "Ya" : 1})
df['eyeDisease'] = df['eyeDisease'].replace({'Tidak' : 0, "Ya" : 1})
df['gadgetSleep'] = df['gadgetSleep'].replace({'Tidak' : 0, "Ya" : 1})
df['genFactor'] = df['genFactor'].replace({"Tidak" : 1, "Kurang Tahu" : 2, "Ya" : 3})
df['nutriFood'] = df['nutriFood'].replace({"Tidak" : 1, "Jarang" : 2, "Ya" : 3})
print(df)

# Ubah tanda baca
df['leftMinusSph'] = pd.to_numeric(df['leftMinusSph'].str.replace(',', '.').astype(float))
df['rightMinusSph'] = pd.to_numeric(df['rightMinusSph'].str.replace(',', '.').astype(float))
df['leftPlusSph'] = pd.to_numeric(df['leftPlusSph'].str.replace(',', '.').astype(float))
df['rightPlusSph'] = pd.to_numeric(df['rightPlusSph'].str.replace(',', '.').astype(float))
df['leftCyl'] = pd.to_numeric(df['leftCyl'].str.replace(',', '.').astype(float))
df['rightCyl'] = pd.to_numeric(df['rightCyl'].str.replace(',', '.').astype(float))

# Penanganan missing value
df['typeOfEyeDisease'] = df['typeOfEyeDisease'].fillna(0)
df['leftMinusSph'] = df['leftMinusSph'].fillna(0)
df['rightMinusSph'] = df['rightMinusSph'].fillna(0)
df['leftPlusSph'] = df['leftPlusSph'].fillna(0)
df['rightPlusSph'] = df['rightPlusSph'].fillna(0)
df['leftCyl'] = df['leftCyl'].fillna(0)
df['rightCyl'] = df['rightCyl'].fillna(0)
df['genFactor'] = df['genFactor'].fillna(1)

# Deteksi outlier
def detectOutlierIQR(data):
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = [x for x in data if x < lower_bound or x > upper_bound]

# Penanganan outlier
def replaceOutliersWithMinMax(data, column_name):
    outliers = detectOutlierIQR(data[column_name])
    Q1 = np.percentile(data[column_name], 25)
    Q3 = np.percentile(data[column_name], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    min_non_outlier = data[column_name][(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)].min()
    max_non_outlier = data[column_name][(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)].max()
    data[column_name] = data[column_name].apply(lambda x: min_non_outlier if x < lower_bound else (max_non_outlier if x > upper_bound else x))

# Penghapusan variabel
df = df.drop(columns=['timestamp', 'name', 'sex', 'darkAct', 'gadgetSleep'])
print(df)

# Pemodelan random forest
X = df.drop(columns=['typeOfEyeDisease', 'leftMinusSph', 'rightMinusSph', 'leftPlusSph', 'rightPlusSph', 'leftCyl', 'rightCyl', 'eyeDisease'])
Y = df.drop(columns=['age', 'gadgetPerHour', 'wrongLens', 'genFactor', 'nutriFood', 'typeOfEyeDisease', 'leftMinusSph', 'rightMinusSph', 'leftPlusSph', 'rightPlusSph', 'leftCyl', 'rightCyl'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

model = RandomForestClassifier()
model.fit(X, Y)

Y_predict = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100
print('Accuracy    :', str(accuracy_score(Y_test, Y_predict) * 100),'%')
print('Sensitivity :', str(sens))
print('Specificity :', str(spec))
print('Precision   :', str(precision_score(Y_test, Y_predict)))
print('Recall      :', str(recall_score(Y_test, Y_predict)))

def predict_eye_disease(age, gadgetPerHour, wrongLens, genFactor, nutriFood):
    data_input = [[age, gadgetPerHour, wrongLens, genFactor, nutriFood]]
    prediction = model.predict(data_input)

    if prediction[0] == 0:
        result_message = "Anda tidak berpotensi memiliki penyakit mata (silinder, minus, dan plus)"
    else:
        result_message = "Anda berpotensi memiliki penyakit mata (silinder, minus, dan plus)"

    return result_message

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gadgetPerHour = int(request.form['gadgetPerHour'])
    wrongLens = int(request.form['wrongLens'])
    genFactor = int(request.form['genFactor'])
    nutriFood = int(request.form['nutriFood'])

    result = predict_eye_disease(age, gadgetPerHour, wrongLens, genFactor, nutriFood)

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)