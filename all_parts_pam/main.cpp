#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/face.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace cv::ml;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

Mat face_detection(Mat img)
{
    CascadeClassifier face_cascade;

    face_cascade.load("/home/kvs/haarcascade_frontalface_alt.xml");

    vector<Rect> faces;

    face_cascade.detectMultiScale( img, faces, 1.11,4,0,Size(40, 40));

    Rect face_pt;

    for( int i = 0; i < faces.size(); i++ )
    {

        face_pt.x = faces[i].x;
        face_pt.y = faces[i].y;
        face_pt.width = (faces[i].width);
        face_pt.height = (faces[i].height);

        Point pt1(faces[i].x, faces[i].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));

        rectangle(img, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    }


    Mat img2=img(face_pt);
    return img2;
}

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


char getCharacterByBrightness(int brightness)
{
    char map[10] = {' ', '.', ',', ':', ';', 'o', 'x', '%', '#', '@'};
    return map[(255-brightness)*10/256];
}

PAM_EXTERN int pam_sm_setcred( pam_handle_t *pamh, int flags, int argc, const char **argv ) {
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    printf("Acct mgmt\n");
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_authenticate( pam_handle_t *pamh, int flags,int argc, const char **argv ) {
    // Get the path to your CSV:
    string fn_haar = "/home/kvs/haarcascade_frontalface_alt.xml";
    string fn_csv = "/home/kvs/training_data.txt";
    int deviceId = 0;

//    // These vectors hold the images and corresponding labels:
//    vector<Mat> images;
//    vector<int> labels;
//    // Read in the data:
//    try {
//        read_csv(fn_csv, images, labels);
//    } catch (cv::Exception& e) {
//        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
//        exit(1);
//    }
//    // Get the height and width from the first image
//    // (we need to reshape incoming faces to this size):
//    int im_width = images[0].cols;
//    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();

//    model->train(images, labels);
//    model->setThreshold(60);

//    model->save("/home/kvs/face_MY_FACE.xml");

     model->read("/home/kvs/face_qwerty.xml");
     model->setThreshold(80);




    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }



    Ptr<SVM> svm1 = StatModel::load<SVM>("/home/kvs/smile_MY_FACE_1000_500.xml");


    cout << "Look at the webcam!" << endl;

    // Holds the current frame from the Video device:
    Mat frame;

    //frame = imread("/home/kvs/test.jpg");

    int count = 1;
    while(count < 300) {
        system("clear");

        cap >> frame;

       //frame = imread("/home/kvs/MUG/neutral/" + std::to_string(count) + ".jpg");
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);

        Mat gray_ASCII;
        //resize(gray, gray_ASCII, Size(204, 78));
        resize(gray, gray_ASCII, Size(204, 84));
        //resize(gray, gray_ASCII, Size(80, 24));

        for(int x = 0; x <gray_ASCII.rows; x++) {
            for(int y = 0; y < gray_ASCII.cols; y++) {
                std::cout << getCharacterByBrightness((int)gray_ASCII.at<uchar>(x,y));
                //std::cout << (int)gray.at<uchar>(x,y) << " ";
            }
            std::cout << std::endl;
        }


        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // Get the faces, make a prediction and
        // annotate it in the video.
        for(unsigned int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image:
            Mat face = gray(face_i);
            // Resizing the face:
            Mat face_resized;
            cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);
            // Perform the prediction:
            int label;
            double dist;
            model->predict(face_resized, label, dist);

            cout << label << " " << dist << endl;



            //Распознавание улыбки


            Mat testimage = tan_triggs_preprocessing(face_resized);
            //Mat testimage = face_resized;
            //imshow("face pre proccessed",testimage);
            //waitKey(0);
            //resize(testimage,testimage,Size(200,200));// resizing into the size of training set
            //imshow("resized",testimage);
            //waitKey(0);
            testimage = testimage.reshape(0,1);

            //reshaping and converting into float

            Mat test=Mat(testimage.size(),CV_32FC1);
            testimage.convertTo(test, CV_32FC1,1.0/255.0);

            //predicting SVM

            float res= svm1->predict(test);

            //assigning response

            cout<<res<<endl;
            res==1? cout<<"smile\n":cout<<"neutral\n";


//            if ((label == 1)&&(res == 1)) {
//                cout << "Success!" << endl;
//            } else {
//                cout << "Failure!" << endl;
//            }


            if ((label == 1)&&(res == 1)) {
                //system("clear");
                cout << "Success!" << endl;
                cap.release();
                cout << "@@@" << cap.isOpened() << "@" << endl;
                return PAM_SUCCESS;

            } else if ((label == 1)&&(res != 1)) {
                //system("clear");
                cout << "Smile, please!" << endl;
            } else if ((label != 1)) {
                //system("clear");
                cout << "Get out, fucking bastard!" << endl;
                return PAM_PERM_DENIED;
            }






        }

//            if (label != -1) {
//                cout << "Success!" << endl;
//                //cap.release();
//                return 0;
//            } else {
//                cout << "Failure!" << endl;
//                return 1;
//            }

//        }



        ++count;
    }
    //system("clear");
    cout << "Failure!" << endl;
    return PAM_PERM_DENIED;
}
