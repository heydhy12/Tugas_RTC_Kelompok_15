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
#include <QtWidgets/QComboBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QLineEdit *xInput;
    QComboBox *functionCombo;
    QComboBox *angleCombo;
    QPushButton *calculateButton;
    QLabel *resultLabel;
    QLabel *termsLabel;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(400, 300);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        xInput = new QLineEdit(centralwidget);
        xInput->setObjectName(QString::fromUtf8("xInput"));

        verticalLayout->addWidget(xInput);

        functionCombo = new QComboBox(centralwidget);
        functionCombo->addItem(QString());
        functionCombo->addItem(QString());
        functionCombo->addItem(QString());
        functionCombo->setObjectName(QString::fromUtf8("functionCombo"));

        verticalLayout->addWidget(functionCombo);

        angleCombo = new QComboBox(centralwidget);
        angleCombo->addItem(QString());
        angleCombo->addItem(QString());
        angleCombo->setObjectName(QString::fromUtf8("angleCombo"));

        verticalLayout->addWidget(angleCombo);

        calculateButton = new QPushButton(centralwidget);
        calculateButton->setObjectName(QString::fromUtf8("calculateButton"));

        verticalLayout->addWidget(calculateButton);

        resultLabel = new QLabel(centralwidget);
        resultLabel->setObjectName(QString::fromUtf8("resultLabel"));

        verticalLayout->addWidget(resultLabel);

        termsLabel = new QLabel(centralwidget);
        termsLabel->setObjectName(QString::fromUtf8("termsLabel"));
        termsLabel->setWordWrap(true);

        verticalLayout->addWidget(termsLabel);

        verticalLayout->setStretch(3, 1);
        verticalLayout->setStretch(4, 1);
        MainWindow->setCentralWidget(centralwidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Taylor Series Calculator", nullptr));
        xInput->setPlaceholderText(QCoreApplication::translate("MainWindow", "Enter value of x", nullptr));
        functionCombo->setItemText(0, QCoreApplication::translate("MainWindow", "sin", nullptr));
        functionCombo->setItemText(1, QCoreApplication::translate("MainWindow", "cos", nullptr));
        functionCombo->setItemText(2, QCoreApplication::translate("MainWindow", "tan", nullptr));

        angleCombo->setItemText(0, QCoreApplication::translate("MainWindow", "radians", nullptr));
        angleCombo->setItemText(1, QCoreApplication::translate("MainWindow", "degrees", nullptr));

        calculateButton->setText(QCoreApplication::translate("MainWindow", "Calculate", nullptr));
        resultLabel->setText(QCoreApplication::translate("MainWindow", "Result:", nullptr));
        termsLabel->setText(QCoreApplication::translate("MainWindow", "Taylor Series:", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
