# Aplikasi Prediksi Harga Mobil Bekas

Aplikasi ini adalah dashboard interaktif berbasis web yang dibangun menggunakan **Streamlit**. Tujuannya untuk memprediksi harga jual mobil bekas berdasarkan berbagai spesifikasi teknis menggunakan model Machine Learning yang telah dilatih sebelumnya.

---

## 1. Arsitektur dan Alur Kerja
Aplikasi mengikuti alur kerja *End-to-End* sederhana:
1.  **Pemuatan Data & Model**: Mengambil dataset referensi (`processes2.csv`) dan model `.joblib`.
2.  **Input Pengguna**: Mengambil spesifikasi mobil melalui antarmuka sidebar.
3.  **Validasi & Preprocessing**: Memastikan input sesuai tipe data dan melakukan *Label Encoding*.
4.  **Prediksi**: Menjalankan data melalui dua model regresi (Random Forest & SVR).
5.  **Output**: Menampilkan hasil prediksi dan estimasi nilai pasar rata-rata.

---

## 2. Fitur Utama

### A. Manajemen Cache (`@st.cache`)
Aplikasi menggunakan decorator caching dari Streamlit untuk meningkatkan performa:
* **`@st.cache_data`**: Digunakan untuk fungsi `load_data()`. Dataset CSV hanya dibaca sekali saat aplikasi pertama kali dijalankan.
* **`@st.cache_resource`**: Digunakan untuk fungsi `load_models()`. Objek model disimpan dalam cache agar tidak perlu dimuat ulang setiap kali ada interaksi.

### B. Input Dinamis & Validasi Ketat
Aplikasi secara otomatis mengkategorikan kolom dataset menjadi:
* **Integer Input**: Untuk nilai bulat seperti `year`, `km_driven`, dan `seats`.
* **Float Input**: Untuk nilai desimal seperti `Mileage` dan `max_power`.
* **Dropdown (Selectbox)**: Untuk kolom kategorikal yang opsinya diambil dari nilai unik dataset.

### C. Penanganan Error
Terdapat blok `try-except` untuk menangani skenario file model rusak (`EOFError`) atau file tidak ditemukan (`FileNotFoundError`).

---

## 3. Pendekatan Machine Learning

### Model yang Digunakan
Aplikasi ini menggunakan pendekatan **Multi-Model Inference**:
1.  **Random Forest Regressor (RFR)**: Algoritma berbasis *Ensemble Learning* yang sangat baik dalam menangani hubungan fitur yang kompleks.
2.  **Support Vector Regressor (SVR)**: Menggunakan prinsip *Support Vector Machines* untuk mencari fungsi regresi yang optimal dalam ruang berdimensi tinggi.

### Preprocessing & Post-processing
* **Label Encoding**: Mengonversi kategori teks menjadi numerik menggunakan `LabelEncoder` sebelum masuk ke model.
* **Inverse Scaling**: Mengembalikan hasil prediksi yang ternormalisasi ke harga asli menggunakan rata-rata (`mean`) dan standar deviasi (`std`) dari kolom `selling_price`.
* **Ensemble Averaging**: Memberikan estimasi nilai pasar akhir dengan mengambil rata-rata dari hasil prediksi RFR dan SVR.

---

## 4. Cara Menjalankan
1. Pastikan file `app.py`, `processes2.csv`, dan kedua file `.joblib` berada dalam folder yang sama.
2. Instal dependensi: `pip install streamlit pandas numpy scikit-learn joblib`.
3. Jalankan perintah: `streamlit run app.py`.
