#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QtConcurrent>
#include <QRect>
#include <QDesktopWidget>
#include <QMessageBox>

#include <QMovie>

#include <QDebug>
#include "LoggingCategories.h"

//#include "ImagePreprocessing.h"

#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/face.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace cv::ml;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    timer(this),
    cap(0),
    capture_flag(false),
    flag_neutral(false),
    flag_smile(false),
    counter(0),
    number_of_images(600)
{
    ui->setupUi(this);

    QRect position = frameGeometry();
    position.moveCenter(QDesktopWidget().availableGeometry().center());
    move(position.topLeft());

    QMovie *movie = new QMovie("loader_new.gif");
    ui->label_3->setMovie(movie);
    movie->setSpeed(1000);
    movie->start();
    ui->label_3->setVisible(false);

    if(!cap.isOpened()) {
        QMessageBox::critical(this, "ERROR", "Unable to open webcam! \nExiting program!\n");
        qCritical(logCritical()) << "Unable to open webcam!";
        exit(1);
    } else {
        qInfo(logInfo()) << "Webcam has been turned on successfully";
    }

    string fn_haar = "/home/kvs/haarcascade_frontalface_alt.xml";
    if (!haar_cascade.load(fn_haar)) {
        QMessageBox::critical(this, "ERROR", "Unable to read haar cascade! \nExiting program!\n");
        qCritical(logCritical()) << "Unable to read haar cascade!";
        exit(1);
    } else {
        qInfo(logInfo()) << "Haar cascade has been loaded successfully";
    }

    connect(&timer, SIGNAL(timeout()), this, SLOT(on_timeout()));
    timer.start(2);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_timeout()
{
    Mat frame;
    cap >> frame;
    // Convert the current frame to grayscale:
    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);
    equalizeHist(gray, gray);
    // Find the faces in the frame:
    vector< Rect_<int> > faces;
    haar_cascade.detectMultiScale(gray, faces, CV_HAAR_FIND_BIGGEST_OBJECT);
    if (faces.empty()) {
        // Convert an image for Qt:
        cvtColor(frame, frame, CV_BGR2RGB);
        // Display an image in label:
        ui->label->setPixmap(QPixmap::fromImage(QImage(frame.data, frame.cols,
                                                       frame.rows, frame.step,
                                                       QImage::Format_RGB888)));
    } else {
        // Get the faces:
        for(unsigned int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Draw a green rectangle around the detected face:
            rectangle(frame, face_i, CV_RGB(0, 255,0), 1);
        }
        // Convert an image for Qt:
        cvtColor(frame, frame, CV_BGR2RGB);
        // Display an image in label:
        ui->label->setPixmap(QPixmap::fromImage(QImage(frame.data, frame.cols,
                                                       frame.rows, frame.step,
                                                       QImage::Format_RGB888)));
        // Save only certain number of face images:

        if (flag_neutral) {
            if (capture_flag && (counter < (2*number_of_images/4))) {
                ui->label_3->show();

                // Crop the face from the image:
                Mat face = gray(faces[0]);
                // Resizing the face:
                Mat face_resized;
                cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);

                // Save images for models:
                images.push_back(face_resized);
                // Save labels for face recognition model:
                labels_for_face_model.push_back(1);
                // Save labes for smile recognition model:
                labels_for_smile_model.push_back(0);

                ++counter;
            } else if (capture_flag && !(counter < (2*number_of_images/4))) {
                capture_flag = false;
                flag_neutral = false;
                counter = 0;
                ui->label_2->setText("Image capturing has been !");
                qInfo(logInfo()) << "Finish image capturing (neutral)";
                ui->pushButton_2->setEnabled(true);
                ui->pushButton->setEnabled(false);
                ui->pushButton_3->setEnabled(false);
                ui->label_3->setVisible(false);

            }

        } else if (flag_smile) {
            if (capture_flag && (counter < (2*number_of_images/4))) {
                ui->label_3->show();

                // Crop the face from the image:
                Mat face = gray(faces[0]);
                // Resizing the face:
                Mat face_resized;
                cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);

                // Save images for models:
                images.push_back(face_resized);
                // Save labels for face recognition model:
                labels_for_face_model.push_back(1);
                // Save labes for smile recognition model:
                labels_for_smile_model.push_back(1);

                ++counter;
            } else if (capture_flag && !(counter < (2*number_of_images/4))) {
                capture_flag = false;
                flag_smile = false;
                counter = 0;
                ui->label_2->setText("Image capture has been done!");
                qInfo(logInfo()) << "Finish image capturing (smile)";
                ui->pushButton_3->setEnabled(true);
                ui->pushButton->setEnabled(false);
                ui->pushButton_2->setEnabled(false);
                ui->label_3->setVisible(false);
            }
        }
    }
}

void MainWindow::on_pushButton_clicked()
{
    ui->pushButton->setEnabled(false);
    capture_flag = true;
    flag_neutral = true;
    ui->label_2->setText("Image capture in progress...");
    qInfo(logInfo()) << "Start image capturing (neutral)";
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->pushButton_2->setEnabled(false);
    capture_flag = true;
    flag_smile = true;
    ui->label_2->setText("Image capture in progress...");
    qInfo(logInfo()) << "Start image capturing (smile)";
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->pushButton_3->setEnabled(false);
    QFuture<void> future = QtConcurrent::run(this, &MainWindow::save_models);
}

void MainWindow::save_models()
{
    ui->label_2->setText("Model training in progress...");
    ui->label_3->show();

    // Create a Face Recognizer and train it on the given images:
    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    qInfo(logInfo()) << "Start training face recognition model";
    model->train(images, labels_for_face_model);
    qInfo(logInfo()) << "Finish training face recognition model";
    model->setThreshold(60);
    qInfo(logInfo()) << "Start saving face recognition model";
    model->save("/home/kvs/face_qwerty_test.xml");
    qInfo(logInfo()) << "Finish saving face recognition model";

    // Create a Smile Recognizer and train it on the given images:
    Ptr<FaceRecognizer> model2 = LBPHFaceRecognizer::create();
    qInfo(logInfo()) << "Start training smile recognition model";
    model2->train(images, labels_for_smile_model);
    qInfo(logInfo()) << "Finish training smile recognition model";
    model2->setThreshold(1000);
    qInfo(logInfo()) << "Start saving smile recognition model";
    model2->save("/home/kvs/smile_qwerty_test.xml");
    qInfo(logInfo()) << "Finish saving smile recognition model";

    ui->label_2->setText("Models training has been done! Close the app!");
    ui->pushButton_3->setEnabled(false);
    ui->label_3->setVisible(false);
}
