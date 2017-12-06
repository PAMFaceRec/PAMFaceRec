#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>

#include <syslog.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "opencv2/face.hpp"

//#include "ImagePreprocessing.h"

using namespace std;
using namespace cv;
using namespace cv::face;
using namespace cv::ml;

char get_character_by_brightness(int brightness)
{
    char map[10] = {' ', '.', ',', ':', ';', 'o', 'x', '%', '#', '@'};
    return map[(255 - brightness) * 10 / 256];
}

PAM_EXTERN int pam_sm_setcred( pam_handle_t *pamh, int flags, int argc, const char **argv ) {
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    printf("Acct mgmt\n");
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_authenticate( pam_handle_t *pamh, int flags,int argc, const char **argv ) {

    openlog("pam_face_authentication", LOG_ODELAY, LOG_AUTH);

    // Open webcam:
    VideoCapture cap(0);
    if(!cap.isOpened()) {
        cerr << "Webcam cannot be opened." << endl;
        syslog( LOG_AUTH|LOG_CRIT, "Webcam cannot be opened.");
        return PAM_AUTHINFO_UNAVAIL;
    }

    // Load Haar cascade:
    string fn_haar = "/home/kvs/haarcascade_frontalface_alt.xml";
    CascadeClassifier haar_cascade;
    if (!haar_cascade.load(fn_haar)) {
        cerr << "Haar cascade cannot be loaded." << endl;
        syslog( LOG_AUTH|LOG_CRIT, "Haar cascade cannot be loaded.");
        return PAM_AUTHINFO_UNAVAIL;
    }

    // Load face recognition model:
    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read("/home/kvs/face_qwerty_test.xml");
    model->setThreshold(150);

    // Load smile recognition model:
    Ptr<FaceRecognizer> model2 = LBPHFaceRecognizer::create();
    model2->read("/home/kvs/smile_qwerty_test.xml");

    cout << "Look at the webcam!" << endl;

    // Holds the current frame from the Video device:
    Mat frame;
    int count = 1;

    while(count < 300) {
        // Clear console:
        system("clear");

        // Get the current frame:
        cap >> frame;

        // Clone the current frame:
        Mat original = frame.clone();

        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);

        // Create Mat for ASCII visualisation:
        Mat gray_ASCII;
        resize(gray, gray_ASCII, Size(204, 75));

        // Create a buffer for ASCII symbols:
        std::stringstream ss;

        for(int x = 0; x <gray_ASCII.rows; x++) {
            for(int y = 0; y < gray_ASCII.cols; y++) {
                ss << get_character_by_brightness((int)gray_ASCII.at<uchar>(x,y));
            }
            ss << std::endl;
        }

        // Output ASCII-symbols from the buffer:
        cout << ss.str();

        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // Get the faces, make a prediction and
        // annotate it in the video:
        for(unsigned int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image:
            Mat face = gray(face_i);
            // Resizing the face:
            Mat face_resized;
            cv::resize(face, face_resized, Size(200, 200), 1.0, 1.0, INTER_CUBIC);

            // Face recognition:
            int label;
            double dist;
            model->predict(face_resized, label, dist);

            // Show predicted label and distance:
            cout << label << " " << dist << endl;

            // Smile recognition:
            int res;
            double dist2;
            model2->predict(face_resized, res, dist2);

            // Show predicted emotion:
            if (res == 1) {
                cout<<"smile" << endl;
            } else {
                cout<<"neutral" << endl;
            }

            if ((label == 1) && (res == 1)) {
                system("clear");
                cout << "Success!" << endl;
                return PAM_SUCCESS;
            } else if ((label == 1) && (res != 1)) {
                cout << "Smile, please!" << endl;
            } else if ((label != 1)) {
                system("clear");
                cout << "Failure!" << endl;
                return PAM_PERM_DENIED;
            }
        }
        ++count;
    }
    cout << "Failure!" << endl;
    return PAM_PERM_DENIED;
}
