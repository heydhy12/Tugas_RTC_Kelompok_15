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
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QComboBox *functionComboBox;
    QComboBox *angleComboBox;
    QDoubleSpinBox *xInput;
    QPushButton *calculateButton;
    QPlainTextEdit *resultText;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(400, 300);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        functionComboBox = new QComboBox(centralWidget);
        functionComboBox->addItem(QString());
        functionComboBox->addItem(QString());
        functionComboBox->addItem(QString());
        functionComboBox->setObjectName(QString::fromUtf8("functionComboBox"));

        verticalLayout->addWidget(functionComboBox);

        angleComboBox = new QComboBox(centralWidget);
        angleComboBox->addItem(QString());
        angleComboBox->addItem(QString());
        angleComboBox->setObjectName(QString::fromUtf8("angleComboBox"));

        verticalLayout->addWidget(angleComboBox);

        xInput = new QDoubleSpinBox(centralWidget);
        xInput->setObjectName(QString::fromUtf8("xInput"));
        xInput->setMinimum(-9999.000000000000000);
        xInput->setMaximum(9999.000000000000000);
        xInput->setDecimals(4);
        xInput->setValue(0.000000000000000);

        verticalLayout->addWidget(xInput);

        calculateButton = new QPushButton(centralWidget);
        calculateButton->setObjectName(QString::fromUtf8("calculateButton"));

        verticalLayout->addWidget(calculateButton);

        resultText = new QPlainTextEdit(centralWidget);
        resultText->setObjectName(QString::fromUtf8("resultText"));
        resultText->setReadOnly(true);

        verticalLayout->addWidget(resultText);

        MainWindow->setCentralWidget(centralWidget);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        functionComboBox->setItemText(0, QCoreApplication::translate("MainWindow", "sin", nullptr));
        functionComboBox->setItemText(1, QCoreApplication::translate("MainWindow", "cos", nullptr));
        functionComboBox->setItemText(2, QCoreApplication::translate("MainWindow", "tan", nullptr));

        angleComboBox->setItemText(0, QCoreApplication::translate("MainWindow", "deg", nullptr));
        angleComboBox->setItemText(1, QCoreApplication::translate("MainWindow", "rad", nullptr));

        calculateButton->setText(QCoreApplication::translate("MainWindow", "Hitung", nullptr));
        (void)MainWindow;
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
