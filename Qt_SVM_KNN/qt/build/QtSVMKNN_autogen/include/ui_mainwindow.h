/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.15.13
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QLineEdit *filePathEdit;
    QPushButton *browseButton;
    QPushButton *runButton;
    QHBoxLayout *horizontalLayout_2;
    QLabel *svmAccuracyLabel;
    QLabel *knnAccuracyLabel;
    QHBoxLayout *horizontalLayout_3;
    QLabel *svmPlotLabel;
    QLabel *knnPlotLabel;
    QTableWidget *predictionsTable;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1000, 700);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        filePathEdit = new QLineEdit(centralwidget);
        filePathEdit->setObjectName(QString::fromUtf8("filePathEdit"));

        horizontalLayout->addWidget(filePathEdit);

        browseButton = new QPushButton(centralwidget);
        browseButton->setObjectName(QString::fromUtf8("browseButton"));

        horizontalLayout->addWidget(browseButton);

        runButton = new QPushButton(centralwidget);
        runButton->setObjectName(QString::fromUtf8("runButton"));

        horizontalLayout->addWidget(runButton);


        verticalLayout->addLayout(horizontalLayout);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        svmAccuracyLabel = new QLabel(centralwidget);
        svmAccuracyLabel->setObjectName(QString::fromUtf8("svmAccuracyLabel"));

        horizontalLayout_2->addWidget(svmAccuracyLabel);

        knnAccuracyLabel = new QLabel(centralwidget);
        knnAccuracyLabel->setObjectName(QString::fromUtf8("knnAccuracyLabel"));

        horizontalLayout_2->addWidget(knnAccuracyLabel);


        verticalLayout->addLayout(horizontalLayout_2);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        svmPlotLabel = new QLabel(centralwidget);
        svmPlotLabel->setObjectName(QString::fromUtf8("svmPlotLabel"));

        horizontalLayout_3->addWidget(svmPlotLabel);

        knnPlotLabel = new QLabel(centralwidget);
        knnPlotLabel->setObjectName(QString::fromUtf8("knnPlotLabel"));

        horizontalLayout_3->addWidget(knnPlotLabel);


        verticalLayout->addLayout(horizontalLayout_3);

        predictionsTable = new QTableWidget(centralwidget);
        predictionsTable->setObjectName(QString::fromUtf8("predictionsTable"));
        predictionsTable->setColumnCount(4);
        predictionsTable->setRowCount(0);

        verticalLayout->addWidget(predictionsTable);

        MainWindow->setCentralWidget(centralwidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "SVM and KNN Analysis", nullptr));
        browseButton->setText(QCoreApplication::translate("MainWindow", "Browse...", nullptr));
        runButton->setText(QCoreApplication::translate("MainWindow", "Run Analysis", nullptr));
        svmAccuracyLabel->setText(QCoreApplication::translate("MainWindow", "SVM Accuracy: ", nullptr));
        knnAccuracyLabel->setText(QCoreApplication::translate("MainWindow", "KNN Accuracy: ", nullptr));
        predictionsTable->setHorizontalHeaderLabels(QStringList()
            << QCoreApplication::translate("MainWindow", "Sample", nullptr)
            << QCoreApplication::translate("MainWindow", "SVM Prediction", nullptr)
            << QCoreApplication::translate("MainWindow", "KNN Prediction", nullptr)
            << QCoreApplication::translate("MainWindow", "Actual", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
