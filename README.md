📚 **TUGAS RTC KELOMPOK 15**

GitHub repositori ini berisi seluruh hasil pengerjaan mata kuliah **Rekayasa Teknologi Cerdas** oleh Kelompok 15 yang berfokus terhadap **implementasi algoritma menggunakan Bahasa Rust dan pengembangan GUI menggunakan QT**
Dataset dari website : _**https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/discussion/491365**_

**🍒Daftar anggota kelompok:**
1. Monica Intan Wijayanti (2042221014)
2. Alan Darmawan Dewantoro (2042221020)
3. Heydhy Maulana Syahputra (2042221095)

**🍒 Terdapat 5 project, dengan Deskripsi project  dan logika step running nya sebagai berikut:**
1. **Implementasi fungsi sinx dan cosx menggunakan pendekatan Deret Taylor**
    - Program dibuat dengan mengolah rumus pendekatan deret taylor
    - User melakukan input fungsi nilai sudut
    - Inisialisasi
    - Pehitungan deret taylor untuk program
    - Program menampilkan hasil output perhitungan

2. **Implementasi fungsi sinx dan cosx menggunakan pendekatan Look-up Table**
    - Program dibuat dengan input table look-up sebagai pendekatan perhitungan nilai sinx dan cosx
    - User melakukan input fungsi nilai sudut
    - Inisialisasi
    - Pehitungan look-up table untuk program
    - Program menampilkan hasil output perhitungan

3. **Impementasi SVM (Support Vector Machine) dan kNN (k-Nearest Neighbors) menggunakan dataset dengan tema Industrial Automation**
    - Program membaca dataset
    - Pembagian data set dengan training 80% dan testing 20%
    - Training model oleh program
    - Testing model oleh program untuk melakukan prediksi
    - User memasukkan input untuk menghitung akurasi model
    - Program akan menampilkan akurasi dan hasil prediksi

4. **Implementasi algoritma NN (Neural Network) menggunakan dataset dengan tema Industrial Automation**
    - Program membaca dataset
    - Pembagian data set dengan training 80% dan testing 20%
    - Mendefinisikan struktur NN dengan hidden layer 5 & 3
    - Training model oleh program
    - Testing model oleh program untuk melakukan prediksi
    - User memasukkan input untuk menghitung akurasi model
    - Program akan menampilkan akurasi dan hasil prediksi

5. **Pengembahan Frontend GUI menggunakan QT sebagai visualisasi**
    - GUI start
    - User memberikan input data
    - Forntend dan Backend Connection
    - program menerima hasil prediksi dari model
    - Tampilan hasil prediksi ke GUI berupa anga, plot tabel, dan grafik

**🍒 Hasil:**
Dari percobaan yang telah dilakukan oleh penulis dengan data set berjumlah 1383, dimana 80% data sebagai training dan 20% data sebagai testing, telah berhasil dilakukan pembuatan program deret taylor, look-up table, SVM dan kNN, serta Neural Network  menggunakan software Rust. Disini, dapat dinilai bahwa proses pembacaan program menggunakan Rust lebih cepat dibandingkan dengan phyton. Rust memakan waktu sekitar  12 detik, sedangkan phyton memakan waktu sekitar 13 menit. Nilai akurasi yang di dapatkan dalam program ini adalah 96% untuk training dan 95% untuk testing dengan 4 input (air temperature, process temperature, rational speed, dan torque) dan 4 output (power failure, overstrain failure, no failure, heat dissipation failure). Tugas ini menggunakan 40 jurnal internasional, dengan 10 jurnal sebagai sitasi laporan dan 30 jurnal sebagai referensi.

**Link laporan:**
https://its.id/m/RTC_LAPORANKELOMPOK15

**Link PPT:**
https://its.id/m/POWERPOINT_KELOMPOK15RTC

**Link Referensi 30 Jurnal Internasional:**
https://its.id/m/30JurnalReferensiIEEE 
