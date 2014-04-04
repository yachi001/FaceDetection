#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <stdio.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat visual_orientation(Mat ori, Size winSize, int cellSize, float scale) {

    //Initialize
    Mat result(Size(ori.cols, ori.rows), CV_8UC3);
    result = Scalar(0, 0, 0);

    int col = winSize.width / cellSize;
    int row = winSize.height / cellSize;

    float temp[2] = { 0 };

    float*** ori_vec = new float**[row];
    for (int i = 0; i < row; i++) {
        ori_vec[i] = new float*[col];
        for (int j = 0; j < col; j++) {
            ori_vec[i][j] = new float[2];
        }
    }

    //Get average V(x,y) for each cell
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            for (int k = 0; k < cellSize; k++) {
                for (int p = 0; p < cellSize; p++) {
                    temp[0] += ori.at<Vec2f>(i * cellSize + k, j * cellSize + p)[0];
                    temp[1] += ori.at<Vec2f>(i * cellSize + k, j * cellSize + p)[1];
                }
            }
            ori_vec[i][j][0] = temp[0] / (float)(cellSize * cellSize);
            ori_vec[i][j][1] = temp[1] / (float)(cellSize * cellSize);
            temp[0] = 0;
            temp[1] = 0;
            //rm = rm > ori_vec[i][j][0]? rm : ori_vec[i][j][0];
            //im = im > ori_vec[i][j][1]? im : ori_vec[i][j][1];
        }
    }

    //Draw
    for (int y = 0; y < row; y++)
    for (int x = 0; x < col; x++) {

        if (ori_vec[y][x][0] == 0 && ori_vec[y][x][1] == 0)
            continue;

        int draw_x = x * cellSize;
        int draw_y = y * cellSize;
        int m_x = draw_x + cellSize / 2;
        int m_y = draw_y + cellSize / 2;

        float x1 = m_x - ori_vec[y][x][0] * scale;
        float y1 = m_y - ori_vec[y][x][1] * scale;
        float x2 = m_x + ori_vec[y][x][0] * scale;
        float y2 = m_y + ori_vec[y][x][1] * scale;

        line(result, Point(x1, y1), Point(x2, y2), CV_RGB(255, 255, 255));
    }

    return result;
}

Mat orientationVector(Mat src, float thresh_strength) {
    Mat display;

    /// Gradient X
	Mat grad_x;
    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    //convertScaleAbs( grad_x, display );
    //imshow( "x", display );

    /// Gradient Y
	Mat grad_y;
    Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
    //convertScaleAbs( grad_y, display );
    //imshow( "y", display );

    /// Edge Strength
	Mat strength;
    grad_x.convertTo(grad_x, CV_32F);
    grad_y.convertTo(grad_y, CV_32F);
    magnitude(grad_x, grad_y, strength);
    threshold(strength, strength, thresh_strength, 0, THRESH_TOZERO);
    //convertScaleAbs(strength, display);
    //imshow( "magnitude", display);

    /// Edge direction
	Mat direction;
    divide(grad_y, grad_x, direction);
    Mat dir_mod;
    direction.copyTo(dir_mod);

    for (int i = 0; i < direction.rows; i++) {
        for (int j = 0; j < direction.cols; j++){
            direction.at<float>(i, j) = atan(direction.at<float>(i, j)) + PI / 2;
            if (direction.at<float>(i, j) >= PI
                && direction.at<float>(i, j) < (PI * 2)) {
                direction.at<float>(i, j) = direction.at<float>(i, j) - PI;
            }
            else if (direction.at<float>(i, j) >= 0
                && direction.at<float>(i, j) < PI){
            }
            else
                cout << "direction value out of range\n";
            dir_mod.at<float>(i, j) = fmod(direction.at<float>(i, j), (float)PI);
        }
    }
    //convertScaleAbs(direction, display, (255.0 / PI));
    //imshow( "direction", display);

    ///Oritentation
	Mat orientation(src.rows, src.cols, CV_32FC2, Scalar(0, 0));
    for (int i = 0; i < src.rows; i++)
    for (int j = 0; j < src.cols; j++) {
        orientation.at<Vec2f>(i, j)[0] = strength.at<float>(i, j) * cos(direction.at<float>(i, j));
        orientation.at<Vec2f>(i, j)[1] = strength.at<float>(i, j) * sin(direction.at<float>(i, j));
    }
    //convertScaleAbs(visual_orientation(orientation, src.size(), 4, 0.0135), display);
    //imshow("Orientation", display);

    return orientation;
}

double dist(Vec2f vi, Vec2f vm){
    double dist;
    if (norm(vi) > 0 && norm(vm) > 0) {
        dist = norm(vi, vm);
    }
    else {
        dist = 255;
    }
    return dist;
}

//Backup
float dist(float vi1, float vi2, float vm1, float vm2){
    float dist, vi, vm;
    vi = sqrt(vi1 * vi1 + vi2 * vi2);
    vm = sqrt(vm1 * vm1 + vm2 * vm2);
    if (vi > 0 && vm > 0) {
        dist = sqrt(pow((vi1 - vm1), 2) + pow((vi2 - vm2), 2));
    }
    else {
        dist = 255;
    }
    return dist;
}

Mat getDistance(Mat vi, Mat vm) {
    int sz[] = { vi.rows, vi.cols, vm.rows, vm.cols };
    int p[4];
    Mat distance2(4, sz, CV_32F, Scalar::all(255));

    for (int x = 0; x < vi.rows - vm.rows + 1; x++) {
        for (int y = 0; y < vi.cols - vm.cols + 1; y++) {
            for (int m = 0; m < vm.rows; m++) {
                for (int n = 0; n < vm.cols; n++) {
                    p[0] = x;
                    p[1] = y;
                    p[2] = m;
                    p[3] = n;
                    distance2.at<float>(p) = dist(vm.at<Vec2f>(m, n), vi.at<Vec2f>(x + m, y + n));
                }
            }
        }
    }

    return distance2;
}

Mat elasticDist(Mat vi, Mat vm, float w[]){
    Mat cost(vi.rows, vi.cols, CV_32F);
    double min, distTotal;
    float distNeighbor[9] = { 65025 };

    int sz[] = { vi.rows, vi.cols, vm.rows, vm.cols };
    int p[4];
    Mat distance(4, sz, CV_32F, Scalar::all(0));

    distance = getDistance(vi, vm);

    for (int x = 0; x < vi.rows; x++) {
        for (int y = 0; y < vi.cols; y++) {
            distTotal = 0;
            p[0] = x;
            p[1] = y;
            for (int m = 0; m < vm.rows; m++) {
                for (int n = 0; n < vm.cols; n++) {
                    min = -255;
                    /// +w(k,l)
                    p[2] = m; p[3] = n;
                    distNeighbor[4] = w[4] + pow(distance.at<float>(p), (float)2);
                    if (m > 0 && n > 0) {
                        p[2] = m - 1; p[3] = n - 1;
                        distNeighbor[0] = w[0] + pow(distance.at<float>(p), (float)2);
                    }
                    if (n > 0) {
                        p[2] = m; p[3] = n - 1;
                        distNeighbor[1] = w[1] + pow(distance.at<float>(p), (float)2);
                        if (m < vm.rows - 1) {
                            p[2] = m + 1; p[3] = n - 1;
                            distNeighbor[2] = w[2] + pow(distance.at<float>(p), (float)2);
                        }
                    }

                    if (m > 0) {
                        p[2] = m - 1; p[3] = n;
                        distNeighbor[3] = w[3] + pow(distance.at<float>(p), (float)2);
                        if (n < vm.cols - 1) {
                            p[2] = m - 1; p[3] = n + 1;
                            distNeighbor[6] = w[6] + pow(distance.at<float>(p), (float)2);
                        }
                    }
                    if (m < vm.rows - 1) {
                        p[2] = m + 1; p[3] = n;
                        distNeighbor[5] = w[5] + pow(distance.at<float>(p), (float)2);
                    }
                    if (n < vm.cols - 1) {
                        p[2] = m; p[3] = n + 1;
                        distNeighbor[7] = w[7] + pow(distance.at<float>(p), (float)2);
                        if (m < vm.rows - 1) {
                            p[2] = m + 1; p[3] = n + 1;
                            distNeighbor[8] = w[8] + pow(distance.at<float>(p), (float)2);
                        }
                    }
                    ///Find min
                    for (int i = 0; i < 9; i++) {
                        if (min == -255) {
                            min = distNeighbor[i];
                        }
                        else {
                            min = min < distNeighbor[i] ? min : distNeighbor[i];
                        }
                    }//end-i(min)
                    distTotal += sqrt(min);
                    //cout << sqrt(min) << "\t";
                }//end-n
            }//end-m
            cost.at<float>(x, y) = distTotal;
            //cout << distTotal << "\t";
        }//end-y
    }//end-xs


    return cost;
}

void printMinMax(string matName, Mat mat)
{
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);

    cout << "=========  " << matName << "  =========" << endl;
    cout << "min val : " << minVal << endl;
    cout << "max val: " << maxVal << endl;

}
void printMatrix(Mat m)
{
	for( int i=0; i<m.rows; i++ ) {
		for( int j=0; j<m.cols; j++ ) { 
			cout << m.at<float>(i, j) << " ";
		}
		cout << endl;
	}
}

float getMin(Mat mat)
{
	double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
	return minVal;
}

float getMax(Mat mat)
{
	double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
	return maxVal;
}


int main()
{
    cout << "Debugging" << endl;

    ///Image
    Mat src, src_gray;
    Mat avg, avg_gray;
    Mat ori_src(src.rows, src.cols, CV_32FC2, Scalar(0, 0));
    Mat ori_avg(avg.rows, avg.cols, CV_32FC2, Scalar(0, 0));

    double resize_factor;
    float w[9] = { 96, 64, 96, 64, 0, 64, 96, 64, 96 };

    /// Load an image
    src = imread("test1.jpg");
    avg = imread("m(01-32).jpg");

    if (!src.data || !avg.data) {
        return -1;
    }

	Mat src_blur;
    GaussianBlur(src, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);

    /// Scale the average face to fit
    if (avg.rows > src.rows) {
        resize_factor = (double)src.rows / (double)avg.rows;
        resize(avg, avg, Size(), resize_factor, resize_factor);
    }
    if (avg.cols > src.cols) {
        resize_factor = (double)src.cols / (double)avg.cols;
        resize(avg, avg, Size(), resize_factor, resize_factor);
    }

    /// Scale all
    resize(avg, avg, Size(), 0.4, 0.4);
    //resize(src, src, Size(), 0.2, 0.2);

    /// Convert it to gray
    cvtColor(src_blur, src_gray, CV_RGB2GRAY);
    cvtColor(avg, avg_gray, CV_RGB2GRAY);

    /// Get orientations
    Mat display;
    ori_src = orientationVector(src_gray, 10);
    convertScaleAbs(visual_orientation(ori_src, ori_src.size(), 4, 0.0135), display);
    imshow("Orientation - Image", display);

    ori_avg = orientationVector(avg_gray, 10);
    convertScaleAbs(visual_orientation(ori_avg, ori_avg.size(), 4, 0.0135), display);
    imshow("Orientation - Model", display);
    cout << "Done orientation\n";

    /// Distance
    Mat distance;
    distance = elasticDist(ori_src, ori_avg, w);

	const float threshold = 0;
	const float min = getMin(distance);
	const float max = getMax(distance);
	
	for( int i=0; i<distance.rows; i++ ) {
		for( int j=0; j<distance.cols; j++ ) {
			distance.at<float>(i, j) = (distance.at<float>(i, j) - min) / (max-min) ; //normalize
			if( distance.at<int>(i, j) > threshold ) {
				rectangle(src, Point(i, j), Point(i-avg.rows, j-avg.cols), Scalar(0, 0, 255));

			}
		}
	}

	imshow("elastic distance", distance);
	imshow("rect", src);
    
    cout << "Done all\n";
    waitKey(0);
    return 0;
}


