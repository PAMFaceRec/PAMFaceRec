#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 8) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/im-folder> </path/to/csv-file> <user-id> <im_width> <im_height> <device-id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/im-folder> -- Path to the folder where face images will be saved." << endl;
        cout << "\t </path/to/csv-file> -- Path to the CSV file." << endl;
        cout << "\t <user-id> -- User ID in database." << endl;
        cout << "\t <im_width> -- Width of face image." << endl;
        cout << "\t <im_height> -- Height of face image." << endl;
        cout << "\t <device-id> -- Device ID." << endl;
        exit(1);
    }
    // Get the path to your CSV:
    string fn_haar = string(argv[1]);
    string fn_im_folder = string(argv[2]);
    string fn_csv = string(argv[3]);
    int userId = atoi(argv[4]);
    int im_width = atoi(argv[5]);
    int im_height = atoi(argv[6]);
    int deviceId = atoi(argv[7]);

    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    int count = 1;

    // Open file  for writing:
    std::fstream fs;
    fs.open(fn_csv, std::fstream::out);
    if(!fs.is_open()) {
        cerr << "cvs-file " << fn_csv << "cannot be opened." << endl;
        return -1;
    }
    while(true) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        equalizeHist(gray, gray);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces, CV_HAAR_FIND_BIGGEST_OBJECT);
        // Get the faces, show and save it:
        for(unsigned int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            // Show the result:
            imshow("face_recognizer", original);
            // Crop the face from the image:
            Mat face = gray(face_i);
            // Resizing the face:
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Save resized images of face:
            imwrite(fn_im_folder + "/" + std::to_string(count) + ".jpg",face_resized);
            // Write to csv file:
            fs << (fn_im_folder + "/" + std::to_string(count) + ".jpg") << ";" << std::to_string(userId) << endl;
            ++count;
        }
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if (key == 27) {
            break;
        }
    }
    fs.close();
    return 0;
}
