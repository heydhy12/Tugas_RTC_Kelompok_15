#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QLocale>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    // Set default values
    ui->epochSpinBox->setValue(1000);
    ui->plotLabel->setScaledContents(true);
    
    // Configure double spin boxes
    QList<QDoubleSpinBox*> spinBoxes = {
        ui->airTempInput,
        ui->processTempInput,
        ui->rotationalSpeedInput,
        ui->torqueInput
    };
    
    // Force US locale
    QLocale locale(QLocale::English, QLocale::UnitedStates);
    foreach (QDoubleSpinBox* spinBox, spinBoxes) {
        spinBox->setLocale(locale);
        spinBox->setMinimum(0.0);
        spinBox->setMaximum(9999.99);
        spinBox->setSingleStep(0.1);
        spinBox->setValue(0.0);
        spinBox->setDecimals(2);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_browseCsvButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, 
        "Select CSV File", 
        QCoreApplication::applicationDirPath(), 
        "CSV Files (*.csv)");
        
    if (!file.isEmpty()) {
        csvFilePath = file;
        ui->csvPathLabel->setText(QFileInfo(file).fileName());
        
        // Debug output
        qDebug() << "Selected CSV file:" << csvFilePath;
    }
}


void MainWindow::on_trainButton_clicked()
{   
    if (csvFilePath.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please select a CSV file first");
        return;
    }

    int epochs = ui->epochSpinBox->value();
    QString plotPath = "training_plot.png";  
    QString modelPath = "trained_model.bin"; 

    // Add accuracy variable
    double accuracy = 0.0;

    // Panggil fungsi Rust (modified to return accuracy)
    bool success = train_industry_model(
        csvFilePath.toStdString().c_str(),
        epochs,
        plotPath.toStdString().c_str(),
        modelPath.toStdString().c_str(),
        &accuracy  // Pass pointer to accuracy variable
    );

    if (success) {
        QString message = QString("Model trained successfully!\nAccuracy: %1%")
                            .arg(accuracy * 100, 0, 'f', 2);
        QMessageBox::information(this, "Success", message);
        
        ui->accuracyLabel->setText(QString("Training Accuracy: %1%").arg(accuracy * 100, 0, 'f', 2));
        
        displayTrainingPlot();
    } else {
        QMessageBox::critical(this, "Error", "Failed to train model");
    }
}


void MainWindow::on_predictButton_clicked()
{
    QString modelPath = "trained_model.bin";
    if (!QFile::exists(modelPath)) {
        QMessageBox::warning(this, "Warning", "Please train the model first");
        return;
    }

    // Debug input values
    qDebug() << "Prediction Inputs (Raw):";
    qDebug() << "Air Temp:" << ui->airTempInput->value();
    qDebug() << "Process Temp:" << ui->processTempInput->value();
    qDebug() << "Rotational Speed:" << ui->rotationalSpeedInput->value();
    qDebug() << "Torque:" << ui->torqueInput->value();

    // Call Rust function
    PredictionResult* result = predict_failure_type(
        ui->airTempInput->value(),
        ui->processTempInput->value(),
        ui->rotationalSpeedInput->value(),
        ui->torqueInput->value(),
        modelPath.toStdString().c_str()
    );

    if (result == nullptr) {
        QMessageBox::critical(this, "Error", "Prediction failed");
        return;
    }

    // Debug prediction output
    qDebug() << "Prediction Result:";
    qDebug() << "Class:" << result->class_;
    qDebug() << "Probabilities:";
    qDebug() << "Power Failure:" << result->probabilities[0];
    qDebug() << "Overstrain:" << result->probabilities[1];
    qDebug() << "No Failure:" << result->probabilities[2];
    qDebug() << "Heat Dissipation:" << result->probabilities[3];

    // Validasi probabilitas
    if (result->probabilities[0] < 0.0 || result->probabilities[0] > 1.0 ||
        result->probabilities[1] < 0.0 || result->probabilities[1] > 1.0 ||
        result->probabilities[2] < 0.0 || result->probabilities[2] > 1.0 ||
        result->probabilities[3] < 0.0 || result->probabilities[3] > 1.0) {
        QMessageBox::warning(this, "Warning", "Invalid probabilities detected. Model may be corrupted.");
        free_prediction_result(result);
        return;
    }

    // Format probabilities
    QString probabilities = QString(
        "Power Failure: %1%\n"
        "Overstrain Failure: %2%\n"
        "No Failure: %3%\n"
        "Heat Dissipation Failure: %4%")
        .arg(result->probabilities[0] * 100, 0, 'f', 2)
        .arg(result->probabilities[1] * 100, 0, 'f', 2)
        .arg(result->probabilities[2] * 100, 0, 'f', 2)
        .arg(result->probabilities[3] * 100, 0, 'f', 2);
    
    ui->probabilitiesLabel->setText(probabilities);
    
    free_prediction_result(result);
}

void MainWindow::displayTrainingPlot()
{
    QPixmap pixmap("training_plot.png");
    if (!pixmap.isNull()) {
        ui->plotLabel->setPixmap(pixmap.scaled(ui->plotLabel->size(), Qt::KeepAspectRatio));
    } else {
        ui->plotLabel->setText("Training plot not available");
    }
}