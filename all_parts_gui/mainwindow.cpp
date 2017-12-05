#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QtConcurrent>
#include <QRect>
#include <QDesktopWidget>

#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/face.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace cv::ml;

//tan_triggs
Mat tan_triggs_preprocessing(InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2) {

    // Convert to floating point:
    Mat X = src.getMat();
    X.convertTo(X, CV_32FC1);
    // Start preprocessing:
    Mat I;
    pow(X, gamma, I);
    // Calculate the DOG Image:
    {
        Mat gaussian0, gaussian1;
        // Kernel Size:
        int kernel_sz0 = (3*sigma0);
        int kernel_sz1 = (3*sigma1);
        // Make them odd for OpenCV:
        kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
        kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
        GaussianBlur(I, gaussian0, Size(kernel_sz0,kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
        GaussianBlur(I, gaussian1, Size(kernel_sz1,kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
        subtract(gaussian0, gaussian1, I);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(abs(I), alpha, tmp);
            meanI = mean(tmp).val[0];

        }
        I = I / pow(meanI, 1.0/alpha);
    }

    {
        double meanI = 0.0;
        {
            Mat tmp;
            pow(min(abs(I), tau), alpha, tmp);
            meanI = mean(tmp).val[0];
        }
        I = I / pow(meanI, 1.0/alpha);
    }

    // Squash into the tanh:
    {
        Mat exp_x, exp_negx;
    exp( I / tau, exp_x );
    exp( -I / tau, exp_negx );
    divide( exp_x - exp_negx, exp_x + exp_negx, I );
        I = tau * I;
    }
    return I;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    timer(this),
    cap(0),
    capture_flag(false),
    counter(0),
    number_of_images(100)
{
    ui->setupUi(this);
    QRect position = frameGeometry();
    position.moveCenter(QDesktopWidget().availableGeometry().center());
    move(position.topLeft());

    string fn_haar = "/home/kvs/haarcascade_frontalface_alt.xml";
    haar_cascade.load(fn_haar);

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
        // Save only 20 face images:
        if (capture_flag && (counter < (number_of_images / 2))) {
            // Crop the face from the image:
            Mat face = gray(faces[0]);
            // Resizing the face:
            Mat face_resized;
            cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);

            // Save images and labels for face recognition model:
            images.push_back(face_resized);
            labels.push_back(1);

            // Save images and labes for smile recognition model:
            Mat face_resized_f = tan_triggs_preprocessing(face_resized);
            Mat face_resized_f_reshape = face_resized_f.reshape(0, 1);
            trainingImages.push_back(face_resized_f_reshape);
            if (flag_neutral == true) {
                trainingLabels.push_back(0);
            } else if (flag_smile == true) {
                trainingLabels.push_back(1);
            }

            ++counter;
        } else if (capture_flag && !(counter < (number_of_images / 2))) {
            ui->label_2->setText("Image capture has been done!");
            capture_flag = false;

            if (flag_neutral == true) {
                ui->pushButton_2->setEnabled(true);
                ui->pushButton->setEnabled(false);
                ui->pushButton_3->setEnabled(false);
            } else if (flag_smile == true) {
                ui->pushButton_3->setEnabled(true);
                ui->pushButton->setEnabled(false);
                ui->pushButton_2->setEnabled(false);
            }

            flag_neutral = false;
            flag_smile = false;
            counter = 0;          
        }
    }
}

void MainWindow::on_pushButton_clicked()
{
    capture_flag = true;
    flag_neutral = true;
    ui->label_2->setText("Image capture in progress...");
}

void MainWindow::on_pushButton_2_clicked()
{
    capture_flag = true;
    flag_smile = true;
    ui->label_2->setText("Image capture in progress...");
}

void MainWindow::on_pushButton_3_clicked()
{
    QFuture<void> future = QtConcurrent::run(this, &MainWindow::save_models);
}

void MainWindow::save_models()
{
    ui->label_2->setText("Model training in progress...");
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->setThreshold(60);
    model->save("/home/kvs/face_qwerty_test.xml");

    // Create a SVM Smile Recognizer and train it on the given images:
    Mat classes;
    Mat trainingData;
    Mat tra_set(number_of_images,40000,CV_32FC1);
    Mat(trainingImages).copyTo(trainingData);
    trainingData = trainingData.reshape(0,number_of_images);
    trainingData.convertTo(tra_set, CV_32FC1,1.0/255.0);
    Mat(trainingLabels).copyTo(classes);

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    //svm->setNu(0.09);
    //svm->setC(1);
    //svm->setGamma('auto');
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(tra_set, ROW_SAMPLE, classes);
    svm->save("/home/kvs/smile_qwerty_test.xml");
    ui->label_2->setText("Model training has been done!");
}
