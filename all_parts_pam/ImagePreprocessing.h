#ifndef IMAGEPREPROCESSING
#define IMAGEPREPROCESSING

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat tan_triggs_preprocessing(InputArray src,
        float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
        int sigma1 = 2);

#endif // IMAGEPREPROCESSING

