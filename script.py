import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.model_selection as model_selection
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None

# Menggunakan data yang telah dipisahkan sebelumnya, X dan Y
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import pandas as pd

data = pd.read_csv('Mata Rabun dan Silinder.csv')
df = pd.DataFrame(data)
print(df)

# Mengkategorikan nilai data
df['sex'] = df['sex'].replace({'Laki-laki' : 0, "Perempuan" : 1})
df['eyeDisease'] = df['eyeDisease'].replace({'Tidak' : 0, "Ya" : 1})
df['typeOfEyeDisease'] = df['typeOfEyeDisease'].replace({"Rabun Jauh (Mata Minus)" : 1, "Rabun Dekat (Mata Plus)" : 2, "Mata Silinder" : 3, "Rabun Jauh (Mata Minus), Mata Silinder" : 4, "Rabun Dekat (Mata Plus), Mata Silinder" : 5, "Rabun Jauh (Mata Minus), Rabun Dekat (Mata Plus), Mata Silinder" : 6,})
df['wrongLens'] = df['wrongLens'].replace({'Tidak' : 0, "Ya" : 1})
df['darkAct'] = df['darkAct'].replace({'Tidak' : 0, "Ya" : 1})
df['eyeDisease'] = df['eyeDisease'].replace({'Tidak' : 0, "Ya" : 1})
df['gadgetSleep'] = df['gadgetSleep'].replace({'Tidak' : 0, "Ya" : 1})
df['genFactor'] = df['genFactor'].replace({"Tidak" : 1, "Kurang Tahu" : 2, "Ya" : 3})
df['nutriFood'] = df['nutriFood'].replace({"Tidak" : 1, "Jarang" : 2, "Ya" : 3})

# Mengubah tanda koma (,) menjadi tanda titik (.)
df['leftMinusSph'] = pd.to_numeric(df['leftMinusSph'].str.replace(',', '.').astype(float))
df['rightMinusSph'] = pd.to_numeric(df['rightMinusSph'].str.replace(',', '.').astype(float))
df['leftPlusSph'] = pd.to_numeric(df['leftPlusSph'].str.replace(',', '.').astype(float))
df['rightPlusSph'] = pd.to_numeric(df['rightPlusSph'].str.replace(',', '.').astype(float))
df['leftCyl'] = pd.to_numeric(df['leftCyl'].str.replace(',', '.').astype(float))
df['rightCyl'] = pd.to_numeric(df['rightCyl'].str.replace(',', '.').astype(float))

# Deteksi missing value
print(df.isna().sum())

# Penanganan missing value
# Mengubah sekaligus mengelompokkan responden yang tidak memiliki rabun atau silinder ke tipe 0
df['typeOfEyeDisease'] = df['typeOfEyeDisease'].fillna(0)

# Mengubah missing value pada mata minus, plus, dan silinder menjadi 0 (tidak terkena)
## Mata minus
df['leftMinusSph'] = df['leftMinusSph'].fillna(0)
df['rightMinusSph'] = df['rightMinusSph'].fillna(0)

## Mata plus
df['leftPlusSph'] = df['leftPlusSph'].fillna(0)
df['rightPlusSph'] = df['rightPlusSph'].fillna(0)

## Mata silinder
df['leftCyl'] = df['leftCyl'].fillna(0)
df['rightCyl'] = df['rightCyl'].fillna(0)

# Memasukkan responden dengan mata sehat ke kelompok genFactor 1 (tidak)
df['genFactor'] = df['genFactor'].fillna(1)

print(df.isna().sum())

outliers = []

# Deteksi outlier dengan IQR
def detectOutlierIQR(data):
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = [x for x in data if x < lower_bound or x > upper_bound]
  return outliers

# Deteksi outlier dengan Z-Score
def detectOutlierZscore(data):
  threshold = 3
  mean = np.mean(data)
  std = np.std(data)
  outliers = [y for y in data if np.abs((y-mean)/std)>threshold]
  return outliers

# Menampilkan outlier dengan IQR
outlier1 = detectOutlierIQR(df['age'])
print("outlier kolom Age: ",outlier1)
print("banyak outlier Age: ",len(outlier1))
print()

outlier2 = detectOutlierIQR(df['gadgetPerHour'])
print("outlier kolom gadgetPerHour: ",outlier2)
print("banyak outlier gadgetPerHour: ",len(outlier2))
print()

outlier11 = detectOutlierIQR(df['leftMinusSph'])
print("outlier kolom leftMinusSph: ",outlier11)
print("banyak outlier leftMinusSph: ",len(outlier11))
print()

outlier12 = detectOutlierIQR(df['rightMinusSph'])
print("outlier kolom rightMinusSph: ",outlier12)
print("banyak outlier rightMinusSph: ",len(outlier12))
print()

outlier13 = detectOutlierIQR(df['leftPlusSph'])
print("outlier kolom leftPlusSph: ",outlier13)
print("banyak outlier leftPlusSph: ",len(outlier13))
print()

outlier14 = detectOutlierIQR(df['rightPlusSph'])
print("outlier kolom rightPlusSph: ",outlier14)
print("banyak outlier rightPlusSph: ",len(outlier14))
print()

outlier15 = detectOutlierIQR(df['leftCyl'])
print("outlier kolom leftCyl: ",outlier15)
print("banyak outlier leftCyl: ",len(outlier15))
print()

outlier16 = detectOutlierIQR(df['rightCyl'])
print("outlier kolom rightCyl: ",outlier16)
print("banyak outlier rightCyl: ",len(outlier16))
print()

# Deteksi outlier selain kolom 'age' dan 'gadgetPerHour'
outlier3 = detectOutlierZscore(df['sex'])
print("outlier kolom sex: ",outlier3)
print("banyak outlier sex: ",len(outlier3))
print()

outlier4 = detectOutlierZscore(df['typeOfEyeDisease'])
print("outlier kolom typeOfEyeDisease: ",outlier4)
print("banyak outlier typeOfEyeDisease: ",len(outlier4))
print()

outlier5 = detectOutlierZscore(df['wrongLens'])
print("outlier kolom wrongLens: ",outlier5)
print("banyak outlier wrongLens: ",len(outlier5))
print()

outlier6 = detectOutlierZscore(df['darkAct'])
print("outlier kolom darkAct: ",outlier6)
print("banyak outlier darkAct: ",len(outlier6))
print()

outlier7 = detectOutlierZscore(df['eyeDisease'])
print("outlier kolom eyeDisease: ",outlier7)
print("banyak outlier eyeDisease: ",len(outlier7))
print()

outlier8 = detectOutlierZscore(df['gadgetSleep'])
print("outlier kolom gadgetSleep: ",outlier8)
print("banyak outlier gadgetSleep: ",len(outlier8))
print()

outlier9 = detectOutlierZscore(df['genFactor'])
print("outlier kolom genFactor: ",outlier9)
print("banyak outlier genFactor: ",len(outlier9))
print()

outlier10 = detectOutlierZscore(df['nutriFood'])
print("outlier kolom nutriFood: ",outlier10)
print("banyak outlier nutriFood: ",len(outlier10))
print()

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
    return data

df = replaceOutliersWithMinMax(df, 'gadgetPerHour')
outlier2 = detectOutlierIQR(df['gadgetPerHour'])
print("outlier kolom gadgetPerHour: ",outlier2)
print("banyak outlier gadgetPerHour: ",len(outlier2))
print()

# Penghapusan variabel input yang tidak memiliki korelasi dengan variabel output
df = df.drop(columns=['timestamp', 'name', 'sex', 'darkAct', 'gadgetSleep'])

X = df.drop(columns=['typeOfEyeDisease', 'leftMinusSph', 'rightMinusSph', 'leftPlusSph', 'rightPlusSph', 'leftCyl', 'rightCyl', 'eyeDisease'])
Y = df.drop(columns=['age', 'gadgetPerHour', 'wrongLens', 'genFactor', 'nutriFood', 'typeOfEyeDisease', 'leftMinusSph', 'rightMinusSph', 'leftPlusSph', 'rightPlusSph', 'leftCyl', 'rightCyl'])
print(" Data Input ".center(55, "="))
print(X)
print(" Data Output ".center(55, "="))
print(Y)
print("=======================================================")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

model = RandomForestClassifier()
model.fit(X_train, Y_train)
print(X_test)


RandomForestClassifier()

# Classification report Random Forest
Y_predict = model.predict(X_test)
print(" Classification Report Random Forest ".center(55, '='))
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
print("=======================================================")

# Confusion Matrix Random Forest
cr = confusion_matrix(Y_test, Y_predict)
cr_display = ConfusionMatrixDisplay(confusion_matrix = cr, display_labels = ['No', 'Yes'])
cr_display.plot()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Melakukan pemisahan data ke dalam training dan testing
# Ganti X dan Y dengan data input dan output yang sesuai
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Membuat model RandomForestClassifier
model = RandomForestClassifier()

# Melatih model dengan data training
model.fit(X_train, Y_train)

# Fungsi untuk memprediksi berdasarkan input pengguna
def predict_eye_disease(age, gadgetPerHour, wrongLens, genFactor, nutriFood):
    # Lakukan prediksi berdasarkan data input pengguna
    data_input = [[age, gadgetPerHour, wrongLens, genFactor, nutriFood]]
    prediction = model.predict(data_input)

    if prediction[0] == 0:
        return "Anda tidak berpotensi memiliki penyakit mata (silinder, minus, dan plus)"
    else:
        return "Anda berpotensi memiliki penyakit mata (silinder, minus, dan plus)"

# # Meminta input dari pengguna
# age = int(input("Masukkan usia Anda: "))
# gadgetPerHour = int(input("Masukkan jumlah penggunaan gadget per jam: "))
# wrongLens = int(input("Apakah pernah menggunakan lensa yang salah? (0 untuk tidak, 1 untuk ya): "))
# genFactor = int(input("Apakah ada riwayat faktor genetik? (1 untuk tidak, 2 untuk kurang tahu, 3 untuk ya): "))
# nutriFood = int(input("Apakah Anda mengonsumsi makanan bernutrisi? (1 untuk tidak, 2 untuk jarang, 3 untuk ya): "))

# # Melakukan prediksi berdasarkan input pengguna
# result = predict_eye_disease(age, gadgetPerHour, wrongLens, genFactor, nutriFood)

# # Menampilkan hasil prediksi
# if result == 0:
#     print("Anda tidak memiliki penyakit mata.")
# else:
#     print("Anda memiliki penyakit mata.")
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Route untuk halaman utama (index.html)
@app.route('/')
def index():
    return render_template('eyes.html')

# Route untuk menerima prediksi dari form
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan nilai dari form
    age = int(request.form['age'])
    gadgetPerHour = int(request.form['gadgetPerHour'])
    wrongLens = int(request.form['wrongLens'])
    genFactor = int(request.form['genFactor'])
    nutriFood = int(request.form['nutriFood'])

    # Lakukan prediksi dengan model yang sudah dilatih sebelumnya
    result = predict_eye_disease(age, gadgetPerHour, wrongLens, genFactor, nutriFood)


    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
