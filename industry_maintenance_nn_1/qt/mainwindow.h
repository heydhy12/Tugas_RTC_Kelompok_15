#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QPixmap>
#include <QLibrary>  // Tambahkan ini

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_trainButton_clicked();
    void on_predictButton_clicked();
    void on_browseCsvButton_clicked();

private:
    Ui::MainWindow *ui;
    QString csvFilePath;
    void displayTrainingPlot();  // Hapus parameter
};

extern "C" {
    bool train_industry_model(const char* csv_path, int epochs, const char* plot_path, const char* model_path);
    struct PredictionResult {
        int class_;  
        double* probabilities;  
    };
    PredictionResult* predict_failure_type(double air_temp, double process_temp, double rotational_speed, double torque, const char* model_path);
    void free_prediction_result(PredictionResult* result);
}

#endif // MAINWINDOW_H