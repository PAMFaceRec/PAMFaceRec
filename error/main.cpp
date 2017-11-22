#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <security/pam_appl.h>
#include <security/pam_modules.h>
#include "testclass.h"

///////////////////


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/////////////////////


/* expected hook */
PAM_EXTERN int pam_sm_setcred( pam_handle_t *pamh, int flags, int argc, const char **argv ) {
    return PAM_SUCCESS;
}

PAM_EXTERN int pam_sm_acct_mgmt(pam_handle_t *pamh, int flags, int argc, const char **argv) {
    printf("Acct mgmt\n");
    return PAM_SUCCESS;
}

/* expected hook, this is where custom stuff happens */
PAM_EXTERN int pam_sm_authenticate( pam_handle_t *pamh, int flags,int argc, const char **argv ) {
    int retval;

    const char* pUsername;
    retval = pam_get_user(pamh, &pUsername, "Username: ");

    printf("Welcome %s\n", pUsername);

    //----------------------------------------------------------------------------------------------------------------------
    VideoCapture cap(0);
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << 0 << "cannot be opened." << endl;
        return -1;
    }


    // Initialize the inbuilt Harr Cascade frontal face detection
    // Below mention the path of where your haarcascade_frontalface_alt2.xml file is located

     CascadeClassifier face_cascade;
     face_cascade.load( "/home/haarcascade_frontalface_alt.xml" );

    for (;;)
    {

         // Image from camera to Mat

        Mat img;
        cap >> img;
        imwrite("/tmp/tst.jpg", img);
        break;

        // obtain input image from source
        cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);

        // Just resize input image if you want
        resize(img, img, Size(1000,640));

        // Container of faces
         vector<Rect> faces;


         // Detect faces
         face_cascade.detectMultiScale( img, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(140, 140) );


         //Show the results
         // Draw circles on the detected faces

       for( int i = 0; i < faces.size(); i++ )
       {
           Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
           ellipse( img, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
       }

        // To draw rectangles around detected faces
       /* for (unsigned i = 0; i<faces.size(); i++)
                   rectangle(img,faces[i], Scalar(255, 0, 0), 2, 1);*/


        imshow("wooohooo", img);
        int key2 = waitKey(20);

    }

    //----------------------------------------------------------------------------------------------------------------------

    if (retval != PAM_SUCCESS) {
        return retval;
    }

    if (strcmp(pUsername, "backdoor") != 0) {
        return PAM_AUTH_ERR;
    }

    return PAM_SUCCESS;
}
