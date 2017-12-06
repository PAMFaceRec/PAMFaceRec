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

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void save_models();

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QTimer timer;
    VideoCapture cap;
    CascadeClassifier haar_cascade;

    vector<Mat> images;
    vector<int> labels_for_face_model;
    vector<int> labels_for_smile_model;

    bool capture_flag;
    bool flag_neutral;
    bool flag_smile;

    int counter;
    int number_of_images;
};

#endif // MAINWINDOW_H
