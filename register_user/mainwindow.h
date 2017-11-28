#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>

#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT    

    private slots:
    void on_timeout();
    void on_pushButton_clicked();

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QTimer timer;
    VideoCapture cap;
    CascadeClassifier haar_cascade;
    fstream fs;

    bool capture_flag;
    int counter;
};

#endif // MAINWINDOW_H
