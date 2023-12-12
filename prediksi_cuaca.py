import pandas as pd

data = pd.read_csv("seattleweather.csv")
data.head(100)

import matplotlib.pyplot as plt
data.hist(bins=12, figsize=(10,10))
plt.show()

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

data['date'] = enc.fit_transform(data['date'].values)
data['weather'] = enc.fit_transform(data['weather'].values)

data.head()

data.info()

atr_data = data.drop(columns=['weather', 'date'])
atr_data.head()

cls_data = data['weather']
cls_data.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

xtrain, xtemp, ytrain, ytemp = train_test_split(atr_data, cls_data, test_size=0.5, random_state=42)
tree_data = DecisionTreeClassifier(max_depth=5)
tree_data.fit (xtrain, ytrain)

xval, xtest, yval, ytest = train_test_split(xtemp, ytemp, test_size=0.2, random_state=42)
yval_pred = tree_data.predict(xval)
tree_data.predict(xval)

# Menampilkan akurasi data validasi
accuracy_val = accuracy_score(yval, yval_pred)
print(f'Akurasi pada data validasi: {accuracy_val}')

ytrain_pred = tree_data.predict(xtrain)
tree_data.predict(xtrain)

# Menampilkan akurasi data latih
accuracy_train = accuracy_score(ytrain, ytrain_pred)
print(f'Akurasi pada data latih: {accuracy_train}')

from sklearn.model_selection import cross_val_score

# Menampilkan cross validation
scores = cross_val_score(tree_data, xtrain, ytrain, cv=5)
print(scores)

if accuracy_val > 0.8:
    # Menguji model dengan data uji (testing)
    ytest_pred = tree_data.predict(xtest)

    # Mengukur akurasi model pada data uji (testing)
    accuracy_test = accuracy_score(ytest, ytest_pred)
    print(f'Akurasi pada data uji: {accuracy_test}')

    # Menampilkan laporan klasifikasi
    print('\nLaporan Klasifikasi:\n', classification_report(ytest, ytest_pred, zero_division=1))
else:
    print('Model tidak mencapai akurasi yang memuaskan pada data validasi. Periksa dan optimalkan model Anda.')

from sklearn import metrics
import seaborn as sns

# Menampilkan matrix confusion
cm = metrics.confusion_matrix(ytest, ytest_pred)
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

from sklearn.tree import export_graphviz
from sklearn import tree

# Mendapatkan label kelas yang sesuai dengan transformasi LabelEncoder
class_names = enc.inverse_transform(tree_data.classes_)

# Menggunakan model Decision Tree yang sudah di-fit
export_graphviz(tree_data, out_file='tree_cuaca.dot', class_names=class_names,
                feature_names=atr_data.columns, impurity=False, filled=True)

import graphviz

# Menampilkan tree
with open('tree_cuaca.dot') as fig:
  figsize= (6,6)
  dot_graph = fig.read()
graphviz.Source(dot_graph)

import numpy as np

# Masukan input dari pengguna
def get_user_input():
    precipitation = float(input("Masukkan jumlah presipitasi (%): "))
    temp_max = float(input("Masukkan suhu maksimum (°C): "))
    temp_min = float(input("Masukkan suhu minimum (°C): "))
    wind = float(input("Masukkan kecepatan angin (km/h): "))

    user_input = [precipitation, temp_max, temp_min, wind]
    return user_input

# Menampilkan input dari pengguna
print("\nMasukkan kondisi cuaca untuk diprediksi:")
new_data = [get_user_input()]  # No need for np.array

# Menggunakan model Decision Tree yang sudah di-fit untuk prediksi cuaca
prediction = tree_data.predict(new_data)

# Mendapatkan label cuaca yang sesuai dengan transformasi LabelEncoder
weather_labels = enc.inverse_transform(prediction)

# Menampilkan hasil prediksi
print("\n====================")
print(f"Hasil Prediksi Cuaca: {weather_labels[0]}")
print("====================")
