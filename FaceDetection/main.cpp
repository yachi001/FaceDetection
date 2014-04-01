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

Mat visual_orientation (Mat ori, Size winSize, int cellSize, float scale) {

	//Initialize
	Mat result(Size(ori.cols, ori.rows),CV_8UC3);
	result = Scalar(0,0,0);

	int col = winSize.width / cellSize;
    int row = winSize.height / cellSize;

	float temp[2] = {0};

	float*** ori_vec = new float**[row];
	for(int i = 0; i < row; i++) {
		ori_vec[i] = new float*[col];
		for(int j = 0; j < col; j++) {
			ori_vec[i][j] = new float[2];
		}
	}

	//float rm = 0;
	//float im = 0;

	//Get average V(x,y) for each cell
	for(int i = 0; i < row; i++) {
		for(int j = 0; j < col; j++) {
			for(int k = 0; k < cellSize; k++) {
				for(int p = 0; p < cellSize; p++) { 
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
		  //cout << rm << " and " << im << "\n";

	//Draw
	for(int y = 0; y < row; y++)
		for(int x = 0; x < col; x++) {

			if (ori_vec[y][x][0] == 0 && ori_vec[y][x][1] == 0)
                    continue;

			int draw_x = x * cellSize;
			int draw_y = y * cellSize;
			int m_x = draw_x + cellSize/2;
            int m_y = draw_y + cellSize/2;

			float x1 = m_x - ori_vec[y][x][0] * scale;
            float y1 = m_y - ori_vec[y][x][1] * scale;
            float x2 = m_x + ori_vec[y][x][0] * scale;
            float y2 = m_y + ori_vec[y][x][1] * scale;

			line(result, Point(x1, y1), Point(x2, y2), CV_RGB(255,255,255));
		}

	return result;
}

Mat orientationVector (Mat src, int ddepth, int scale, int delta, float thresh_strength) {
	 /// Generate grad_x and grad_y
	  Mat grad_x, grad_y;
	  Mat strength, direction;
	  Mat display;
	  Mat orientation(src.rows, src.cols, CV_32FC2, Scalar(0, 0));

	  //double minVal; 
	  //double maxVal; 
	  //Point minLoc; 
	  //Point maxLoc;

	  /// Gradient X
	  Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  //convertScaleAbs( grad_x, display );
	  //imshow( "x", display );
	 
	  /// Gradient Y
	  Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  //convertScaleAbs( grad_y, display );
	  //imshow( "y", display );

	  /// Edge Strength
	  grad_x.convertTo(grad_x, CV_32F);
	  grad_y.convertTo(grad_y, CV_32F);
	  magnitude(grad_x, grad_y, strength);
	  threshold(strength, strength, thresh_strength, 0, THRESH_TOZERO);
	  //convertScaleAbs(strength, display);
	  //imshow( "magnitude", display);

	  /// Edge direction
	  divide(grad_y, grad_x, direction);
	  Mat dir_mod;
	  direction.copyTo(dir_mod);

	  for( int i=0; i<direction.rows; i++ ) {
		  for( int j=0; j<direction.cols; j++ ){
			   direction.at<float>(i, j) = atan(direction.at<float>(i, j)) + PI / 2;
			   if (direction.at<float>(i, j) >= PI 
				   && direction.at<float>(i, j) < (PI * 2)) {
				   direction.at<float>(i, j) = direction.at<float>(i, j) - PI; 
			   } else if (direction.at<float>(i, j) >= 0 
				   && direction.at<float>(i, j) < PI){ 
			   } else 
				   cout << "direction value out of range\n";
			   dir_mod.at<float>(i, j) = fmod(direction.at<float>(i, j), (float)PI);
		  }
	  }
	  //minMaxLoc( direction, &minVal, &maxVal, &minLoc, &maxLoc );
	  //convertScaleAbs(direction, display, (255.0 / PI));
	  //imshow( "direction", display);

	  ///Oritentation
	  for(int i = 0; i < src.rows; i++)
		  for(int j = 0; j < src.cols; j++) {
			  orientation.at<Vec2f>(i, j)[0] = strength.at<float>(i,j) * cos(direction.at<float>(i, j));
			  orientation.at<Vec2f>(i, j)[1] = strength.at<float>(i,j) * sin(direction.at<float>(i, j));
		  }
	  //convertScaleAbs(visual_orientation(orientation, src.size(), 4, 0.0135), display);
	  //imshow("Orientation", display);

	  return orientation;
}

double dist (Vec2f vi, Vec2f vm){
	double dist;
	if(norm(vi) > 0 && norm(vm) > 0) {
		dist = norm(vi, vm);
	} else {
		dist = 255;
	}
	return dist;
}

//Backup
float dist (float vi1, float vi2, float vm1, float vm2){
	float dist, vi, vm;
	vi = sqrt(vi1 * vi1 + vi2 * vi2);
	vm = sqrt(vm1 * vm1 + vm2 * vm2);
	if(vi > 0 && vm > 0) {
		dist = sqrt(pow((vi1 - vm1), 2) + pow((vi2 - vm2), 2));
	} else {
		dist = 255;
	}
	return dist;
}

Mat getDistance (Mat vi, Mat vm) {
	int sz[] = {vi.rows, vi.cols, vm.rows, vm.cols};
	int p[4];
	Mat distance(4, sz, CV_32F, Scalar::all(255));

	for(int x = 0; x < vi.rows - vm.rows + 1; x++) {
		for(int y = 0; y < vi.cols - vm.cols + 1; y++) {
			for(int m = 0; m < vm.rows; m++) {
				for(int n = 0; n < vm.cols; n++) {
					p[0] = x;
					p[1] = y;
					p[2] = m;
					p[3] = n;
					distance.at<float>(p) = dist(vm.at<Vec2f>(m, n), vi.at<Vec2f>(x + m, y + n));
					//distance.at<float>(p) = dist(vm.at<Vec2f>(m, n)[0], vm.at<Vec2f>(m, n)[1],
						//vi.at<Vec2f>(x + m, y + n)[0], vi.at<Vec2f>(x + m, y + n)[1]);
					//cout << distance.at<float>(p) << "\n";
				}
			}
		}
	}

	return distance;
}

Mat elasticDist (Mat vi, Mat vm, float w[]){
	Mat cost(vi.rows, vi.cols, CV_32F);
	double min, distTotal;
	float distNeighbor[9] = {65025};
	int sz[] = {vi.rows, vi.cols, vm.rows, vm.cols};
	int p[4];
	Mat distance(4, sz, CV_32F, Scalar::all(0));

	distance = getDistance(vi, vm);
	
	for(int x = 0; x < vi.rows; x++) {
		for(int y = 0; y < vi.cols; y++) {
			distTotal = 0;
			p[0] = x;
			p[1] = y;
			for(int m = 0; m < vm.rows; m++) {
				for(int n = 0; n < vm.cols; n++) {
					min = -255;
					/// +w(k,l)
					p[2] = m; p[3] = n;
					distNeighbor[4] = w[4] + pow(distance.at<float>(p), (float)2);
					if( m > 0 && n > 0) {
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
					for(int i = 0; i < 9; i++) {
						if(min == -255) {
							min = distNeighbor[i];
						} else {
							min = min < distNeighbor[i]? min : distNeighbor[i];
						}
					}//end-i(min)
					distTotal += sqrt(min);
					//cout << sqrt(min) << "\t";
				}//end-n
			}//end-m
			//cout << distTotal << "\t";
		}//end-y
	}//end-xs
	

	return cost;
}

int main() 
{
	//image();

	///Image
	  Mat src, src_gray;
	  Mat avg, avg_gray;
	  Mat ori_src(src.rows, src.cols, CV_32FC2, Scalar(0, 0));
	  Mat ori_avg(avg.rows, avg.cols, CV_32FC2, Scalar(0, 0));
	  char* window_name = "Sobel Demo - Simple Edge Detector";
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;
	  double resize_factor;
	  float w[9] = {96, 64, 96, 64, 0, 64, 96, 64, 96};

	  /// Load an image
	  src = imread( "test1.jpg" );
	  avg = imread("m(01-32).jpg");

	  if( !src.data || !avg.data)
	  { return -1; }

	  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	  /// Scale the average face to fit
	  if(avg.rows > src.rows) {
		  resize_factor = (double)src.rows / (double)avg.rows;
		  resize(avg, avg, Size(), resize_factor, resize_factor);
	  }
	  if(avg.cols > src.cols) {
		  resize_factor = (double)src.cols / (double)avg.cols;
		  resize(avg, avg, Size(), resize_factor, resize_factor);
	  }

	  /// Scale all
	  resize(avg, avg, Size(), 0.4, 0.4);
	  //resize(src, src, Size(), 0.2, 0.2);

	  /// Convert it to gray
	  cvtColor( src, src_gray, CV_RGB2GRAY );
	  cvtColor( avg, avg_gray, CV_RGB2GRAY );

	  /// Get orientations
	  Mat display;
	  ori_src = orientationVector(src_gray, ddepth, scale, delta, 10);
	  convertScaleAbs(visual_orientation(ori_src, ori_src.size(), 4, 0.0135), display);
	  imshow("Orientation - Image", display);

	  ori_avg = orientationVector(avg_gray, ddepth, scale, delta, 10);
	  convertScaleAbs(visual_orientation(ori_avg, ori_avg.size(), 4, 0.0135), display);
	  imshow("Orientation - Model", display);
	  
	  /// Distance
	  Mat distance;
	  distance = elasticDist(ori_src, ori_avg, w);

	  //detection();
	  ////cvThreshold( s, s, 100, 100, CV_THRESH_TRUNC );


	  waitKey(0);
	  return 0;
}

/*
int image() 
{
	CascadeClassifier faceCascade;
    faceCascade.load("haarcascade_frontalface_alt.xml");

	Mat im = imread( "test1.jpg" );
	Mat im_gray;

	cvtColor(im, im_gray, CV_BGR2GRAY);
	equalizeHist(im_gray, im_gray);

	vector<Rect> faces;
	faceCascade.detectMultiScale(im_gray, faces, 1.1, 3, 0, Size(30,30));

    for(int i = 0; i < faces.size(); i++)
    {
        Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        Point pt2(faces[i].x, faces[i].y);
 
        rectangle(im, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
    }
	cout << "Number of people in this frame is: " << faces.size();

	imshow("output", im);

	waitKey(0);
	return 0;
}

int video()
{
    CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_alt.xml");
 
    VideoCapture captureDevice;
    captureDevice.open(0);
 
    Mat captureFrame;
    Mat grayscaleFrame;
 

    while(true)
    {
		// SO: caputre the garbage frames
		while (captureFrame.empty()) {
			captureDevice >> captureFrame;
		}
		captureDevice>>captureFrame;

 
        //convert captured image to gray scale and equalize
        cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
        equalizeHist(grayscaleFrame, grayscaleFrame);

        //create a vector array to store the face found
        std::vector<Rect> faces;
 
        //find faces and store them in the vector array
        face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(30,30));
 
        //draw a rectangle for all found faces in the vector array on the original image
        for(int i = 0; i < faces.size(); i++)
        {
            Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
            Point pt2(faces[i].x, faces[i].y);
 
            rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);
        }
 
        //print the output
        imshow("outputCapture", captureFrame);
 
        //pause for 33ms
        waitKey(33);
    }
 
    return 0;
}

int detection()
{
	  Mat img_1 = imread( "test1.jpg" );
	  Mat img_2 = imread( "face.png" );

	  if( !img_1.data || !img_2.data )
	  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	  //-- Step 1: Detect the keypoints using SURF Detector
	  int minHessian = 400;

	  SurfFeatureDetector detector( minHessian );

	  std::vector<KeyPoint> keypoints_1, keypoints_2;

	  detector.detect( img_1, keypoints_1 );
	  detector.detect( img_2, keypoints_2 );

	  //-- Step 2: Calculate descriptors (feature vectors)
	  SurfDescriptorExtractor extractor;

	  Mat descriptors_1, descriptors_2;

	  extractor.compute( img_1, keypoints_1, descriptors_1 );
	  extractor.compute( img_2, keypoints_2, descriptors_2 );

	  //-- Step 3: Matching descriptor vectors using FLANN matcher
	  FlannBasedMatcher matcher;
	  std::vector< DMatch > matches;
	  matcher.match( descriptors_1, descriptors_2, matches );

	  double max_dist = 0; double min_dist = 100;

	  //-- Quick calculation of max and min distances between keypoints
	  for( int i = 0; i < descriptors_1.rows; i++ )
	  { double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	  }

	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
	  //-- PS.- radiusMatch can also be used here.
	  std::vector< DMatch > good_matches;

	  for( int i = 0; i < descriptors_1.rows; i++ )
	  { if( matches[i].distance < 2*min_dist )
		{ good_matches.push_back( matches[i]); }
	  }

	  //-- Draw only "good" matches
	  Mat img_matches;
	  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
				   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	  //-- Show detected matches
	  imshow( "Good Matches", img_matches );

	  for( int i = 0; i < good_matches.size(); i++ )
	  { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

	  waitKey(0);

	  return 0;
 }
 */


