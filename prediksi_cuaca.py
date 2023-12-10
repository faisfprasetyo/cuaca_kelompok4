from sklearn import tree
import numpy as np

# Fungsi untuk mendapatkan input pengguna
def get_user_input():
    try:
        suhu = float(input("Masukkan suhu (°C): "))
        kelembapan = float(input("Masukkan kelembapan (%): "))
        presipitasi = float(input("Masukkan presipitasi (%): "))
        kecepatan_angin = float(input("Masukkan kecepatan angin (km/jam): "))
        return suhu, kelembapan, presipitasi, kecepatan_angin
    except ValueError:
        print("Masukkan harus berupa angka.")
        return get_user_input()

# Mendapatkan jumlah data latih dari pengguna
print("--------------------")
num_samples = int(input("Masukkan jumlah data latih: "))

# Mengumpulkan data latih dari pengguna
features = []
labels = []

for i in range(num_samples):
    print("--------------------")
    print(f"\nData Latih ke-{i + 1}:")
    suhu, kelembapan, presipitasi, kecepatan_angin = get_user_input()
    cuaca = int(input("Masukkan label cuaca (0: Cerah, 1: Berawan, 2: Hujan): "))
    
    features.append([suhu, kelembapan, presipitasi, kecepatan_angin])
    labels.append(cuaca)

# Membuat model Decision Tree
model = tree.DecisionTreeClassifier()

# Melatih model
model.fit(features, labels)

# Memprediksi cuaca untuk suatu kondisi baru
print("--------------------")
print("\nMasukkan kondisi cuaca untuk diprediksi:")
new_data = np.array([get_user_input()])
prediction = model.predict(new_data)

# Menampilkan hasil prediksi dengan lebih baik
weather_labels = {0: "Cerah", 1: "Berawan", 2: "Hujan"}
predicted_weather = weather_labels.get(prediction[0], "Hasil prediksi tidak valid")

print("\n====================")
print(f"Hasil Prediksi Cuaca: {predicted_weather}")
print("====================")
