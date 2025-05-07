📚 **TUGAS RTC KELOMPOK 15**

GitHub repositori ini berisi seluruh hasil pengerjaan mata kuliah **Rekayasa Teknologi Cerdas** oleh **Kelompok 15 Departemen Teknik Instrumentasi, Institut Teknologi Sepuluh Nopember** yang berfokus terhadap **implementasi algoritma menggunakan Bahasa Rust dan pengembangan GUI menggunakan QT**
Dataset dari website : _**https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/discussion/491365**_

**🍒 Daftar anggota kelompok:**
1. Monica Intan Wijayanti (2042221014)
2. Alan Darmawan Dewantoro (2042221020)
3. Heydhy Maulana Syahputra (2042221095)

**🍒 Dosen pengampu: Ahmad Radhy, S.Si., M.Si**

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
  
**🍒 Dataset Yang Digunakan:**
Dataset yang diambil dari website Kaggle adalah data yang berisi monitoring kondisi mesin , khususnya milling machine. Data ini digunakan untuk memantai kondisi mesin milling agar dapat diperbaiki sebelum benar-benar rusak untuk tujuan engineer melakukan predictive maintenance. Dalam dataset tersebut, terdapat 6 parameter yang disebutkan. Namun, dalam penugasan ini penulis hanya memilih 4 parameter utama untuk diproses menggunakan Neural Network. Parameter tersebut adalah:
1.	Air Temperature  (K) : temperature  udara di sekitar mesin.
2.	Process Temperature  (K) : temperature  internal proses mesin ketika bekerja. Jika temperature  terlalu panas (overheat), dapat menyebabkan terjadinya kerusakan mesin atau aus.
3.	Rotational Speed (RPM) : Kecepatan putar poros (spindle) mesin dalam rotasi per menit). Jika kecepatan terlalu tinggi atau terlalu rendah, dapat menyebabkan pemotongan yang tidak optimal.
4.	Torque (Nm) : Besarnya torsi atau gaya puntir yang terjadi saat mesin bekerja. Torsi tinggi bisa artinya mesin sedang memotong bahan yang keras atau sedang mengalami beban berat.

Ke-4 parameter tersebut digunakan sebagai input yang akan di proses dengan hidden layer 5 dan 3. Kemudian, output yang didapatkan ada 4 pula, yaitu:
1.	No Failure : tidak ada masalah.
2.	Power Failure : kerusakan karena masalah daya listrik.
3.	Heat Degradation Failure : kerusakan karena suhu tinggi.
4.	Overstrain Failure : kerusakan karena tekanan berlebihan.

**🍒 Step pembuatan project Neural Network:**
Penjelasan Alur menghubungkan Back End (Rust) ke Front End (QT):
Pertama-tama, penulis menyiapkan data maintenance dari Kaggle yang telah dilakukan editing karena ada beberapa data yang dapat dapat diproses secara langsung. Data ini kemudian dimasukkan dan di proses ke dalam back end (Rust) dan kemudian akan ditampilkan dengan front end (QT). adapun folder yang terdapat dalam pembuatan tugas ini, yaitu:
1. File : Berisi data CSV beserta hasil training
2. SRC : Berisi 6 file (fungsi masing-masing file akan dijelaskan dibawah)
3. QT : Berisi file QT format phyton

Seperti yang telah disebutkan di atas, berikut ini adalah fungsi utama masing-masing yang terdapat dalam folder SRC, yaitu:
-	Data.rs : File yang berisi tentang pemrosesan data, dan normalized data agar dapat digunakan sebagai training dan testing
-	Model.rs : File yang berisi tentang pemodelan neural network.
-	Plot.rs : File yang berisi codingan contuk melakukan plotting data seperti grafik dan gambar.
-	Utils.rs : File yang berisi codingan untuk input manual terminal sesuai dengan 4 parameter.
-	Main.rs : File ini memuat program secara keseluruhan dari 4 folder yang telah disebutkan di atas. Jadi, dalam folder ini program dari awal pembuatan back end hingga training data dijadikan 1.
-	Lib.rs : File ini adalah versi folder main.rs yang akan digunakan sebagai “jembatan” untuk dipanggil oleh QT dalam programnya. Folder ini nantinya akan menghasilkan folder baru Bernama “lib.so”, yang akan digunakan untuk pemrosesan data lebih lanjut di QT.
Kemudian, folder QT akan memanggil dan memberikan Call Back kepada Rust menggunakan folder lib.so. QT yang digunakan adalah QT dengan Bahasa phyton.

**🍒 Hasil:**
Dari percobaan yang telah dilakukan oleh penulis dengan data set berjumlah 1383, dimana 80% data sebagai training dan 20% data sebagai testing, telah berhasil dilakukan pembuatan program deret taylor, look-up table, SVM dan kNN, serta Neural Network  menggunakan software Rust. Disini, dapat dinilai bahwa proses pembacaan program menggunakan Rust lebih cepat dibandingkan dengan phyton. Rust memakan waktu sekitar  12 detik, sedangkan phyton memakan waktu sekitar 13 menit. Nilai akurasi yang di dapatkan dalam program ini adalah 96% untuk training dan 95% untuk testing dengan 4 input (air temperature, process temperature, rational speed, dan torque) dan 4 output (power failure, overstrain failure, no failure, heat dissipation failure). Tugas ini menggunakan 40 jurnal internasional, dengan 10 jurnal sebagai sitasi laporan dan 30 jurnal sebagai referensi.

**Link laporan:**
https://its.id/m/KELOMPOK15RTCLAPORAN

**Link PPT:**
https://its.id/m/PPTRTCKELOMPOK15

**Link Referensi 30 Jurnal Internasional:**
https://its.id/m/30JurnalReferensiIEEE 

---

IN ENGLISH LANGUAGE

---

📚 **RTC GROUP 15 ASSIGNMENT**

This GitHub repository contains all the work results for the course **Intelligent Technology Engineering** by **Group 15 Department of Instrumentation Engineering, Sepuluh Nopember Institute of Technology**, focusing on the **implementation of algorithms using Rust language and GUI development using QT**.  
Dataset from website: _**https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/discussion/491365**_

**🍒 List of group members:**
1. Monica Intan Wijayanti (2042221014)
2. Alan Darmawan Dewantoro (2042221020)
3. Heydhy Maulana Syahputra (2042221095)

**🍒 Supervisor: Ahmad Radhy, S.Si., M.Si**

**🍒 There are 5 projects, with the project descriptions and running logic steps as follows:**
1. **Implementation of sinx and cosx functions using Taylor Series approach**
    - The program is created by processing the Taylor series approximation formula
    - User inputs the angle value function
    - Initialization
    - Taylor series calculation for the program
    - The program displays the calculation output result

2. **Implementation of sinx and cosx functions using Look-up Table approach**
    - The program is created with an input look-up table as an approximation for calculating sinx and cosx values
    - User inputs the angle value function
    - Initialization
    - Look-up table calculation for the program
    - The program displays the calculation output result

3. **Implementation of SVM (Support Vector Machine) and kNN (k-Nearest Neighbors) algorithms using a dataset with the theme of Industrial Automation**
    - The program reads the dataset
    - Dataset is divided into 80% training and 20% testing
    - Model training by the program
    - Model testing by the program for prediction
    - User inputs to calculate model accuracy
    - The program will display accuracy and prediction results

4. **Implementation of NN (Neural Network) algorithm using a dataset with the theme of Industrial Automation**
    - The program reads the dataset
    - Dataset is divided into 80% training and 20% testing
    - Defining NN structure with 5 & 3 hidden layers
    - Model training by the program
    - Model testing by the program for prediction
    - User inputs to calculate model accuracy
    - The program will display accuracy and prediction results

5. **Frontend GUI development using QT as visualization**
    - GUI starts
    - User provides data input
    - Frontend and Backend Connection
    - The program receives prediction results from the model
    - Displaying prediction results on the GUI in the form of numbers, plot tables, and graphs

**🍒 Dataset Used:**
The dataset taken from the Kaggle website contains data that monitors the condition of a machine, specifically a milling machine. This data is used to monitor the condition of the milling machine so that it can be repaired before actual failure occurs, supporting engineers in implementing predictive maintenance. In this dataset, six parameters are mentioned. However, in this assignment, the author only selected four main parameters to be processed using a Neural Network. These parameters are:
1. Air Temperature (K): The temperature of the air surrounding the machine.
2. Process Temperature (K): The internal temperature of the machine during operation. If the temperature becomes too high (overheats), it may cause machine damage or wear.
3. Rotational Speed (RPM): The rotational speed of the machine's spindle in revolutions per minute. If the speed is too high or too low, it can lead to suboptimal cutting performance.
4. Torque (Nm): The amount of torque or twisting force generated during machine operation. A high torque value may indicate the machine is cutting a hard material or under heavy load.

These four parameters are used as input and processed through two hidden layers with 5 and 3 neurons, respectively. The resulting outputs are four types of machine conditions:
1. No Failure: No issues detected.
2. Power Failure: Failure caused by electrical power problems.
3. Heat Degradation Failure: Failure caused by high temperature.
4. Overstrain Failure: Failure caused by excessive mechanical stress.

**🍒 Steps to create a Neural Networ project:**
Explanation of the Flow connecting the Back End (Rust) to the Front End (QT):
First of all, the author prepares maintenance data from Kaggle that has been edited because there is some data that can be processed directly. This data is then entered and processed into the back end (Rust) and will then be displayed with the front end (QT). The folders contained in the creation of this task are:
1. File : Contains CSV data and training results
2. SRC: Contains 6 files (the functions of each file will be explained below)
3. QT: Contains QT files in python format

As mentioned above, the following are the main functions of each of them contained in the SRC folder, namely:
- Data.rs : Files containing data processing, and normalized data so that it can be used for training and testing
- Model.rs : Files that contain about neural network modeling.
- Plot.rs: Files containing coding to plot data such as graphs and images.
- Utils.rs : A file containing the encoding for the manual input of the terminal according to 4 parameters.
- Main.rs : This file contains the entire program from the 4 folders mentioned above. So, in this folder, the program from the beginning of creating the back end to training data is made 1.
- Lib.rs: This file is the version of the main.rs folder that will be used as a "bridge" to be called by QT in its program. This folder will later generate a new folder named "lib.so", which will be used for further data processing in QT. Then, the QT folder will call and give a Call Back to Rust using lib.so folder. The QT used is QT with python language.
  
**🍒 Results:**
From experiments conducted by the authors using a dataset of 1383 entries, where 80% of the data was used for training and 20% for testing, successful implementation of the Taylor series program, look-up table, SVM and kNN, as well as Neural Network using Rust software was achieved. Here, it can be assessed that the process of reading programs using Rust is faster compared to Python. Rust takes about 12 seconds, whereas Python takes about 13 minutes. The accuracy achieved in this program is 96% for training and 95% for testing with 4 inputs (air temperature, process temperature, rotational speed, and torque) and 4 outputs (power failure, overstrain failure, no failure, heat dissipation failure). This assignment uses 40 international journals, with 10 journals as citations for the report and 30 journals as references.

**Link to report:**
https://its.id/m/KELOMPOK15RTCLAPORAN

**Link to PPT:**
https://its.id/m/PPTRTCKELOMPOK15

**Link to 30 International Journal References:**
https://its.id/m/30JurnalReferensiIEEE
