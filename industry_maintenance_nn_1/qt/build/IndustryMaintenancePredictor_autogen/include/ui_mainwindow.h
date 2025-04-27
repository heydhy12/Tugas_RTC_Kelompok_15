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
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QScrollArea *scrollArea;
    QWidget *scrollAreaWidgetContents;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QPushButton *browseCsvButton;
    QLabel *csvPathLabel;
    QLabel *label;
    QSpinBox *epochSpinBox;
    QPushButton *trainButton;
    QGroupBox *plotGroupBox;
    QVBoxLayout *verticalLayout_2;
    QLabel *plotLabel;
    QLabel *accuracyLabel;
    QGroupBox *groupBox_2;
    QFormLayout *formLayout;
    QLabel *label_2;
    QDoubleSpinBox *airTempInput;
    QLabel *label_3;
    QDoubleSpinBox *processTempInput;
    QLabel *label_4;
    QDoubleSpinBox *rotationalSpeedInput;
    QLabel *label_5;
    QDoubleSpinBox *torqueInput;
    QHBoxLayout *horizontalLayout_2;
    QSpacerItem *horizontalSpacer;
    QPushButton *predictButton;
    QSpacerItem *horizontalSpacer_2;
    QLabel *resultLabel;
    QLabel *probabilitiesLabel;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1000, 800);
        MainWindow->setStyleSheet(QString::fromUtf8("\n"
"        QMainWindow {\n"
"          background-color: #f5f5f5;\n"
"        }\n"
"        QGroupBox {\n"
"          border: 1px solid #ccc;\n"
"          border-radius: 5px;\n"
"          margin-top: 10px;\n"
"          padding-top: 15px;\n"
"          font-weight: bold;\n"
"        }\n"
"        QGroupBox::title {\n"
"          subcontrol-origin: margin;\n"
"          left: 10px;\n"
"          padding: 0 3px;\n"
"        }\n"
"        QPushButton {\n"
"          background-color: #4CAF50;\n"
"          color: white;\n"
"          border: none;\n"
"          padding: 8px 16px;\n"
"          border-radius: 4px;\n"
"          min-width: 100px;\n"
"        }\n"
"        QPushButton:hover {\n"
"          background-color: #45a049;\n"
"        }\n"
"        QLabel {\n"
"          font-size: 12px;\n"
"        }\n"
"        QDoubleSpinBox, QSpinBox {\n"
"          padding: 5px;\n"
"          border: 1px solid #ddd;\n"
"          border-radius: 4px;\n"
"        }\n"
"        QScrollArea {\n"
"          border: non"
                        "e;\n"
"        }\n"
"        QWidget#scrollAreaWidgetContents {\n"
"          background-color: #f5f5f5;\n"
"        }\n"
"      "));
        scrollArea = new QScrollArea(MainWindow);
        scrollArea->setObjectName(QString::fromUtf8("scrollArea"));
        scrollArea->setWidgetResizable(true);
        scrollArea->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QString::fromUtf8("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 998, 798));
        verticalLayout = new QVBoxLayout(scrollAreaWidgetContents);
        verticalLayout->setSpacing(10);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(scrollAreaWidgetContents);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setSpacing(10);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        browseCsvButton = new QPushButton(groupBox);
        browseCsvButton->setObjectName(QString::fromUtf8("browseCsvButton"));
        browseCsvButton->setStyleSheet(QString::fromUtf8("background-color: #2196F3;"));

        horizontalLayout->addWidget(browseCsvButton);

        csvPathLabel = new QLabel(groupBox);
        csvPathLabel->setObjectName(QString::fromUtf8("csvPathLabel"));
        csvPathLabel->setWordWrap(true);
        csvPathLabel->setMinimumWidth(300);

        horizontalLayout->addWidget(csvPathLabel);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        epochSpinBox = new QSpinBox(groupBox);
        epochSpinBox->setObjectName(QString::fromUtf8("epochSpinBox"));
        epochSpinBox->setMinimum(100);
        epochSpinBox->setMaximum(10000);
        epochSpinBox->setValue(1000);

        horizontalLayout->addWidget(epochSpinBox);

        trainButton = new QPushButton(groupBox);
        trainButton->setObjectName(QString::fromUtf8("trainButton"));
        trainButton->setStyleSheet(QString::fromUtf8("padding: 8px; font-weight: bold; background-color: #FF9800;"));

        horizontalLayout->addWidget(trainButton);


        verticalLayout->addWidget(groupBox);

        plotGroupBox = new QGroupBox(scrollAreaWidgetContents);
        plotGroupBox->setObjectName(QString::fromUtf8("plotGroupBox"));
        verticalLayout_2 = new QVBoxLayout(plotGroupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        plotLabel = new QLabel(plotGroupBox);
        plotLabel->setObjectName(QString::fromUtf8("plotLabel"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(plotLabel->sizePolicy().hasHeightForWidth());
        plotLabel->setSizePolicy(sizePolicy);
        plotLabel->setMinimumSize(QSize(1000, 800));
        plotLabel->setMaximumSize(QSize(1000, 800));
        plotLabel->setFrameShape(QFrame::Box);
        plotLabel->setAlignment(Qt::AlignCenter);
        plotLabel->setScaledContents(false);
        plotLabel->setStyleSheet(QString::fromUtf8("\n"
"                        background-color: white;\n"
"                        border: 1px solid #ddd;\n"
"                      "));

        verticalLayout_2->addWidget(plotLabel);

        accuracyLabel = new QLabel(plotGroupBox);
        accuracyLabel->setObjectName(QString::fromUtf8("accuracyLabel"));
        accuracyLabel->setAlignment(Qt::AlignCenter);

        verticalLayout_2->addWidget(accuracyLabel);


        verticalLayout->addWidget(plotGroupBox);

        groupBox_2 = new QGroupBox(scrollAreaWidgetContents);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        formLayout = new QFormLayout(groupBox_2);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_2);

        airTempInput = new QDoubleSpinBox(groupBox_2);
        airTempInput->setObjectName(QString::fromUtf8("airTempInput"));
        airTempInput->setDecimals(2);
        airTempInput->setMinimum(0);
        airTempInput->setMaximum(1000);

        formLayout->setWidget(0, QFormLayout::FieldRole, airTempInput);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_3);

        processTempInput = new QDoubleSpinBox(groupBox_2);
        processTempInput->setObjectName(QString::fromUtf8("processTempInput"));
        processTempInput->setDecimals(2);
        processTempInput->setMinimum(0);
        processTempInput->setMaximum(1000);

        formLayout->setWidget(1, QFormLayout::FieldRole, processTempInput);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_4);

        rotationalSpeedInput = new QDoubleSpinBox(groupBox_2);
        rotationalSpeedInput->setObjectName(QString::fromUtf8("rotationalSpeedInput"));
        rotationalSpeedInput->setDecimals(2);
        rotationalSpeedInput->setMinimum(0);
        rotationalSpeedInput->setMaximum(10000);

        formLayout->setWidget(2, QFormLayout::FieldRole, rotationalSpeedInput);

        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_5);

        torqueInput = new QDoubleSpinBox(groupBox_2);
        torqueInput->setObjectName(QString::fromUtf8("torqueInput"));
        torqueInput->setDecimals(2);
        torqueInput->setMinimum(0);
        torqueInput->setMaximum(100);

        formLayout->setWidget(3, QFormLayout::FieldRole, torqueInput);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        predictButton = new QPushButton(groupBox_2);
        predictButton->setObjectName(QString::fromUtf8("predictButton"));
        predictButton->setStyleSheet(QString::fromUtf8("background-color: #4CAF50;"));
        predictButton->setMinimumSize(QSize(150, 0));

        horizontalLayout_2->addWidget(predictButton);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);


        formLayout->setLayout(4, QFormLayout::SpanningRole, horizontalLayout_2);


        verticalLayout->addWidget(groupBox_2);

        resultLabel = new QLabel(scrollAreaWidgetContents);
        resultLabel->setObjectName(QString::fromUtf8("resultLabel"));
        QFont font;
        font.setPointSize(12);
        font.setBold(true);
        font.setWeight(75);
        resultLabel->setFont(font);
        resultLabel->setAlignment(Qt::AlignCenter);
        resultLabel->setStyleSheet(QString::fromUtf8("padding: 10px;"));

        verticalLayout->addWidget(resultLabel);

        probabilitiesLabel = new QLabel(scrollAreaWidgetContents);
        probabilitiesLabel->setObjectName(QString::fromUtf8("probabilitiesLabel"));
        QFont font1;
        font1.setPointSize(11);
        probabilitiesLabel->setFont(font1);
        probabilitiesLabel->setAlignment(Qt::AlignCenter);
        probabilitiesLabel->setStyleSheet(QString::fromUtf8("padding: 10px;"));

        verticalLayout->addWidget(probabilitiesLabel);

        scrollArea->setWidget(scrollAreaWidgetContents);
        MainWindow->setCentralWidget(scrollArea);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "Industry Maintenance Predictor", nullptr));
        groupBox->setTitle(QCoreApplication::translate("MainWindow", "Training", nullptr));
        browseCsvButton->setText(QCoreApplication::translate("MainWindow", "Browse CSV", nullptr));
        csvPathLabel->setText(QCoreApplication::translate("MainWindow", "No file selected", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Epochs:", nullptr));
        trainButton->setText(QCoreApplication::translate("MainWindow", "Train Model", nullptr));
        plotGroupBox->setTitle(QCoreApplication::translate("MainWindow", "Training Results", nullptr));
        accuracyLabel->setText(QCoreApplication::translate("MainWindow", "Accuracy: -", nullptr));
        groupBox_2->setTitle(QCoreApplication::translate("MainWindow", "Prediction", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "Air Temperature (K):", nullptr));
        label_3->setText(QCoreApplication::translate("MainWindow", "Process Temperature (K):", nullptr));
        label_4->setText(QCoreApplication::translate("MainWindow", "Rotational Speed (rpm):", nullptr));
        label_5->setText(QCoreApplication::translate("MainWindow", "Torque (Nm):", nullptr));
        predictButton->setText(QCoreApplication::translate("MainWindow", "Predict", nullptr));
        resultLabel->setText(QCoreApplication::translate("MainWindow", "Result will appear here", nullptr));
        probabilitiesLabel->setText(QCoreApplication::translate("MainWindow", "Probabilities will appear here", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
