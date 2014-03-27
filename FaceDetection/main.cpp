#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>
 
#include <iostream>
#include <stdio.h>
 
using namespace std;
using namespace cv;
 
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



int main() 
{
	//image();

	  Mat src, src_gray;
	  Mat grad;
	  char* window_name = "Sobel Demo - Simple Edge Detector";
	  int scale = 1;
	  int delta = 0;
	  int ddepth = CV_16S;

	  int c;

	  /// Load an image
	  src = imread( "test1.jpg" );

	  if( !src.data )
	  { return -1; }

	  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

	  /// Convert it to gray
	  cvtColor( src, src_gray, CV_RGB2GRAY );


	  /// Generate grad_x and grad_y
	  Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;
	  Mat strength, direction;

	  /// Gradient X
	  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );
	  imshow( "x", abs_grad_x );

	  /// Gradient Y
	  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );
	  imshow( "y", abs_grad_y );

	  // Edge Strength
	  abs_grad_x.convertTo(abs_grad_x, CV_32F);
	  abs_grad_y.convertTo(abs_grad_y, CV_32F);
	  magnitude(abs_grad_x, abs_grad_y, strength);
	  convertScaleAbs(strength, strength);
	  imshow( "magnitude", strength );

	  //const double PI  = 3.14159265358979;

	  //// Edge direction
	  divide(abs_grad_y, abs_grad_x, direction);
	  imshow( "divide-directions", direction );

	  for( int i=0; i<src.rows; i++ ) {
		  for( int j=0; j<src.cols; j++ ){
			   direction.at<float>(i, j) = atan( abs_grad_y.at<float>(i, j) / abs_grad_x.at<float>(i, j)) ;
			   //cout << direction.at<float>(i, j) << "\t";
		  }
		  cout << endl;
	  }
	  convertScaleAbs( direction, direction );
	  imshow( "loop-directions", direction );


	  //detection();
	  ////cvThreshold( s, s, 100, 100, CV_THRESH_TRUNC );


	  waitKey(0);
	  return 0;
}



