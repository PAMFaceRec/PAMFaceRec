#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <stdlib.h>

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    timer(this),
    cap(0),
    capture_flag(false),
    counter(0)
{
    ui->setupUi(this);

    string fn_haar = string("haarcascade_frontalface_alt.xml");
    haar_cascade.load(fn_haar);

    connect(&timer, SIGNAL(timeout()), this, SLOT(on_timeout()));
    timer.start(10);
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
        if (capture_flag && (counter < 20)) {
            // Crop the face from the image:
            Mat face = gray(faces[0]);
            // Resizing the face:
            Mat face_resized;
            cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);
            // Create directory:
            mkdir("/home/kvs/training_data", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            // Save image as jpg:
            imwrite("/home/kvs/training_data/" + std::to_string(counter) + ".jpg",face_resized);
            // Write to txt file paths to images (it is necessary for training of model):
            fs << ("/home/kvs/training_data/" + std::to_string(counter) + ".jpg") << ";" << 1 << endl;
            ++counter;
        } else {
            capture_flag = false;
            counter = 0;
            fs.close();
        }
    }
}

void MainWindow::on_pushButton_clicked()
{
    capture_flag = true;
    fs.open("/home/kvs/training_data.txt", std::fstream::out);
}
