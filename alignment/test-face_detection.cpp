#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame , int count );

String face_cascade_name = "haarcascade_frontalface_alt.xml";

String left_eye_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
String right_eye_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier left_eye_cascade;
CascadeClassifier right_eye_cascade;

String window_name = "Capture - Face detection";

int main( void ) {
    VideoCapture capture;
    Mat frame;
    int count = 1;

    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) ) { printf("--(!)Error loading face cascade\n"); return -1; };
    if( !left_eye_cascade.load( left_eye_cascade_name ) ) { printf("--(!)Error loading left eye cascade\n"); return -1; };
    if( !right_eye_cascade.load( right_eye_cascade_name ) ) { printf("--(!)Error loading right eye cascade\n"); return -1; };

    // Read the video stream
    capture.open( -1 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while (  capture.read(frame) ) {
        if( frame.empty() ) {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        // Apply the classifier to the frame
        detectAndDisplay( frame , count );
        ++count;

        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }
    return 0;
}

void detectAndDisplay( Mat frame , int count) {
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    // Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    if (faces.size() != 0) {
        Point center( faces[0].x + faces[0].width/2, faces[0].y + faces[0].height/2 );
        ellipse( frame, center, Size( faces[0].width/2, faces[0].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[0] );

        cv::Rect left_rect(cv::Point(0, 0), cv::Size(faceROI.size().width*0.5, faceROI.size().height*0.7));
        cv::Rect right_rect(cv::Point(faceROI.size().width*0.5, 0), cv::Size(faceROI.size().width*0.5, faceROI.size().height*0.7));

        cv::Mat left_faceROI;
        cv::Mat right_faceROI;
        left_faceROI = faceROI(left_rect);
        right_faceROI = faceROI(right_rect);

        vector<cv::Rect> left_eye;
        vector<cv::Rect> right_eye;

        // In each face, detect eyes
        left_eye_cascade.detectMultiScale(left_faceROI, left_eye, 1.05, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(10, 10));
        right_eye_cascade.detectMultiScale(right_faceROI, right_eye, 1.05, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(10, 10));

        Point left_eye_center( faces[0].x + left_eye[0].x + left_eye[0].width/2, faces[0].y + left_eye[0].y + left_eye[0].height/2 );
        int left_eye_radius = cvRound( (left_eye[0].width + left_eye[0].height)*0.25 );
        circle( frame, left_eye_center, left_eye_radius, Scalar( 255, 0, 0 ), 4, 8, 0 );

        Point right_eye_center( faces[0].x + faceROI.size().width*0.5 + right_eye[0].x + right_eye[0].width/2, faces[0].y + right_eye[0].y + right_eye[0].height/2 );
        int right_eye_radius = cvRound( (right_eye[0].width + right_eye[0].height)*0.25 );
        circle( frame, right_eye_center, right_eye_radius, Scalar( 255, 0, 0 ), 4, 8, 0 );

        if ((left_eye.size() != 0) && (right_eye.size() != 0)) {
           double x0 = faces[0].x + left_eye[0].x + left_eye[0].width*0.5;
           double x1 = faces[0].x + faces[0].width/2 + right_eye[0].x + right_eye[0].width*0.5;

           double y0 = faces[0].y + left_eye[0].y + left_eye[0].height*0.5;
           double y1 = faces[0].y + right_eye[0].y + right_eye[0].height*0.5;

           double dist1_x = abs(x0 - x1);
           double x_center;

           if (x1 >= x0) {
               x_center = x0 + dist1_x / 2;
           } else {
               x_center = x1 + dist1_x / 2;
           }

           double dist1_y = abs(y0 - y1);
           double y_center;

           if (y1 >= y0) {
               y_center = y0 + dist1_y / 2;
           } else {
               y_center = y1 + dist1_y / 2;
           }

           double p = x_center;
           double q = y_center;

           double angle = atan((y0 - y1) / (x0 - x1));

           cv::Mat scr = frame_gray;
           cv::Mat dst;

           cv::Point2f ptCp(p, q);

           cv::Mat M = cv::getRotationMatrix2D(ptCp, angle*(180 / M_PI), 1.0);
           cv::warpAffine(scr, dst, M, scr.size(), cv::INTER_CUBIC);

           // Show image before and after rotation
           cv::namedWindow("scr");
           cv::imshow("scr", scr);

           cv::namedWindow("dst");
           cv::imshow("dst", dst);

           // Find coordinates of eyes on rotated image
           double x0_r = (x0 - p)*cos(angle) - (y0 - q)*sin(angle) + p;
           double x1_r = (x1 - p)*cos(angle) - (y1 - q)*sin(angle) + p;

           double y0_r = (x0 - p)*sin(angle) + (y0 - q)*cos(angle) + q;
           double y1_r = (x1 - p)*sin(angle) + (y1 - q)*cos(angle) + q;

           double dist_x = abs(x0_r - x1_r);
           double x_r_center;

           if (x1_r >= x0_r) {
               x_r_center = x0_r + dist_x / 2;
           } else {
               x_r_center = x1_r + dist_x / 2;
           }

           double dist_y = abs(y0_r - y1_r);
           double y_r_center;

           if (y1_r >= y0_r) {
               y_r_center = y0_r + dist_y / 2;
           } else {
               y_r_center = y1_r + dist_y / 2;
           }

           double a = sqrt(dist_x*dist_x + dist_y*dist_y);
           double b = 75;
           double k = b / a;

           cv::Mat dst_res;
           resize(dst, dst_res, Size(dst.cols*k, dst.rows*k));

           double x_r_center_res = x_r_center*k;
           double y_r_center_res = y_r_center*k;

           Rect align_rect;
           align_rect.x = static_cast <int> (x_r_center_res - 0.5*k*faces[0].width);
           align_rect.y = static_cast <int> (y_r_center_res - 0.3*k*faces[0].height);
           align_rect.width = static_cast <int> (k*faces[0].width);
           align_rect.height = static_cast <int> (k*faces[0].height);
           cv::Mat dst_res_face = dst_res(align_rect);
           cv::Mat dst_res_face_res;
           resize(dst_res_face, dst_res_face_res, Size(200, 200));
           cv::imwrite(std::to_string(count) + ".jpg", dst_res_face_res);
           std::cout << "align" << std::endl;

        } else {
            cv::Mat face_without_align_res;
            resize(faceROI, face_without_align_res, Size(200, 200));
            //cv::imwrite(std::to_string(count) + ".jpg", face_without_align_res);
            std::cout << "not align" << std::endl;
        }

    } else {
        std::cout << "not face" << std::endl;
    }
    // Show image
    imshow( window_name, frame );
}
