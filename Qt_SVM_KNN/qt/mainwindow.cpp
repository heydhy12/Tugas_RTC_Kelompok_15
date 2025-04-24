#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QPixmap>

extern "C" {
    char* run_analysis(const char* csv_path);
    void free_string(char* s);
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::on_browseButton_clicked() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open CSV File", "", "CSV Files (*.csv)");
    if (!filePath.isEmpty()) {
        ui->filePathEdit->setText(filePath);
    }
}

void MainWindow::on_runButton_clicked() {
    QString filePath = ui->filePathEdit->text();
    if (filePath.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a CSV file first.");
        return;
    }

    char* result_json = run_analysis(filePath.toUtf8().constData());
    if (result_json) {
        displayResults(QString::fromUtf8(result_json));
        free_string(result_json);
    } else {
        QMessageBox::critical(this, "Error", "Failed to analyze the data.");
    }
}

void MainWindow::displayResults(const QString &jsonResults) {
    QJsonDocument doc = QJsonDocument::fromJson(jsonResults.toUtf8());
    if (doc.isNull()) {
        QMessageBox::critical(this, "Error", "Invalid results format.");
        return;
    }

    QJsonObject obj = doc.object();
    
    // Display accuracies
    double svmAccuracy = obj["svm_accuracy"].toDouble() * 100;
    double knnAccuracy = obj["knn_accuracy"].toDouble() * 100;
    ui->svmAccuracyLabel->setText(QString::number(svmAccuracy, 'f', 2) + "%");
    ui->knnAccuracyLabel->setText(QString::number(knnAccuracy, 'f', 2) + "%");

    // Display plots
    QString svmPlotPath = obj["svm_plot_path"].toString();
    QString knnPlotPath = obj["knn_plot_path"].toString();
    ui->svmPlotLabel->setPixmap(QPixmap(svmPlotPath).scaled(400, 300, Qt::KeepAspectRatio));
    ui->knnPlotLabel->setPixmap(QPixmap(knnPlotPath).scaled(400, 300, Qt::KeepAspectRatio));

    // Display sample predictions
    ui->predictionsTable->setRowCount(0);
    QJsonArray samples = obj["sample_predictions"].toArray();
    for (const QJsonValue &sample : samples) {
        QJsonObject s = sample.toObject();
        int row = ui->predictionsTable->rowCount();
        ui->predictionsTable->insertRow(row);
        
        ui->predictionsTable->setItem(row, 0, new QTableWidgetItem(QString::number(s["sample_id"].toInt())));
        ui->predictionsTable->setItem(row, 1, new QTableWidgetItem(s["svm_prediction"].toString()));
        ui->predictionsTable->setItem(row, 2, new QTableWidgetItem(s["knn_prediction"].toString()));
        ui->predictionsTable->setItem(row, 3, new QTableWidgetItem(s["actual"].toString()));
    }
}