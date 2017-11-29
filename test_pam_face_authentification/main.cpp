#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/face.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

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

PAM_EXTERN int pam_sm_setcred( pam_handle_t *pamh, int flags, int argc, const char **argv ) {
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    printf("Acct mgmt\n");
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_authenticate( pam_handle_t *pamh, int flags,int argc, const char **argv ) {
    int retval;

    const char* pUsername;
    retval = pam_get_user(pamh, &pUsername, "Username: ");

    printf("Welcome %s\n", pUsername);

    // Get the path to your CSV:
    string fn_haar = string("/haarcascade_frontalface_alt.xml");
    string fn_csv = string("/dir_train.txt");
    int deviceId = atoi("0");

    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data:
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    // Get the height and width from the first image
    // (we need to reshape incoming faces to this size):
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();
    model->setThreshold(115);
    model->train(images, labels);
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
    // Holds the current frame from the Video device:
    Mat frame;
    int count;
    while(1000) {
        ++count;
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);
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
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Perform the prediction:
            int prediction = model->predict(face_resized);
//            // Draw a green rectangle around the detected face:
//            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
//            // Create the text we will annotate the box with:
//            string box_text = format("Prediction = %d", prediction);
//            // Calculate the position for annotated text
//            // (make sure we don't put illegal values in there):
//            int pos_x = std::max(face_i.tl().x - 10, 0);
//            int pos_y = std::max(face_i.tl().y - 10, 0);
//            // And now put it into the image:
//            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            cout << prediction << endl;


            if (prediction == 3) {
                cap.release();
                return PAM_SUCCESS;
            } else {
                return PAM_PERM_DENIED;
            }

        }
    }


    //----------------------------------------------------------------------------------------------------------------------

//    if (retval != PAM_SUCCESS) {
//        return retval;
//    }

//    if (strcmp(pUsername, "backdoor") != 0) {
//        return PAM_AUTH_ERR;
//    }

//    return PAM_SUCCESS;
}
