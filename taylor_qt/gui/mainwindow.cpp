#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QLibrary>
#include <QMessageBox>
#include <QString>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->calculateButton, &QPushButton::clicked, 
        this, &MainWindow::onCalculateClicked);


}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::onCalculateClicked()
{
    QString libPath = "/home/heydhy12/taylor_qt/target/release/libtaylor_lookup.so";
    
    QLibrary lib(libPath);
    if (!lib.load()) {
        QMessageBox::critical(this, "Error", 
            "Gagal load library!\n" 
            "Path: " + libPath + "\n"
            "Error: " + lib.errorString());
        return;
    }

    typedef char* (*CalculateFunc)(const char*, const char*, double);
    typedef void (*FreeStringFunc)(char*);

    auto calculate = (CalculateFunc)lib.resolve("calculate");
    auto free_string = (FreeStringFunc)lib.resolve("free_string");

    if (!calculate || !free_string) {
        QMessageBox::critical(this, "Error", "Failed to resolve functions!");
        return;
    }

    char* result = calculate(
        ui->functionComboBox->currentText().toUtf8().constData(),
        ui->angleComboBox->currentText().toUtf8().constData(),
        ui->xInput->value()
    );

    ui->resultText->setPlainText(QString::fromUtf8(result));
    free_string(result);
}