#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QString libPath = "/home/alandarmawan34/taylor/qt/lib/librust_taylor.so";
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
