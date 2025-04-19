#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <dlfcn.h>

extern "C" {
    const char* rust_taylor_series(const char*, double, bool, double*, double*);
    double rust_factorial(unsigned int);
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->calculateButton, &QPushButton::clicked, 
            this, &MainWindow::onCalculateClicked);
}

void MainWindow::onCalculateClicked() {
    void* lib = dlopen("/home/alandarmawan34/taylor/qt/lib/librust_taylor.so", RTLD_LAZY);
    if (!lib) {
        QMessageBox::critical(this, "Error", "Failed to load Rust library:\n" + QString(dlerror()));
        return;
    }

    // Resolve symbol
    auto rust_taylor_series_fn = (const char* (*)(const char*, double, bool, double*, double*)) dlsym(lib, "rust_taylor_series");
    if (!rust_taylor_series_fn) {
        QMessageBox::critical(this, "Error", "Failed to locate rust_taylor_series:\n" + QString(dlerror()));
        dlclose(lib);
        return;
    }

    double result;
    double terms[5];
    QByteArray funcBytes = ui->functionCombo->currentText().toUtf8();
    bool isDegree = ui->angleCombo->currentText() == "degrees";
    double x = ui->xInput->text().toDouble();

    const char* error = rust_taylor_series_fn(
        funcBytes.constData(),
        x,
        isDegree,
        &result,
        terms
    );

    if (error) {
        QMessageBox::warning(this, "Error", error);
    } else {
        ui->resultLabel->setText(QString("Result: %1").arg(result));
        ui->termsLabel->setText(
            QString("Taylor Series:\n%1 + %2h + %3h² + %4h³ + %5h⁴")
            .arg(terms[0], 0, 'f', 4)
            .arg(terms[1], 0, 'f', 4)
            .arg(terms[2], 0, 'f', 4)
            .arg(terms[3], 0, 'f', 4)
            .arg(terms[4], 0, 'f', 4));
    }

    dlclose(lib);
}


MainWindow::~MainWindow() {
    delete ui;
}