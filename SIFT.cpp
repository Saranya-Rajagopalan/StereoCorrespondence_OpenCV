#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


int main(int argc, char** argv)
{

	Mat img_1 = imread("C:/Project1/im2.png", IMREAD_GRAYSCALE);
	Mat img_2 = imread("C:/Project1/im6.png", IMREAD_GRAYSCALE);
	Mat gtimg = imread("C:/Project1/disp2.png", IMREAD_GRAYSCALE);

	if (!img_1.data || !img_2.data || !gtimg.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}
	
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int num = 300;
	Ptr<SIFT> detector = SIFT::create(num);
	
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
	detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

	//------Show Keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1),
	DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1),
	DrawMatchesFlags::DEFAULT);
	imwrite("SIFT_Keypoints1.png", img_keypoints_1);
	imwrite("SIFT_Keypoints2.png", img_keypoints_2);

	Mat DepthImg = Mat::zeros(img_1.size(), CV_8U);
	std::vector< DMatch > matchpairs(descriptors_1.rows);

	//--------Calculate matching using SSD algorithm

	for (int k1 = 0; k1 < descriptors_1.rows; ++k1)		//descriptor in img1
	{
		double minSSD = 10000000;
		for (int k2 = 0; k2 < descriptors_2.rows; ++k2)		//descriptor in img2
		{
			double sum = 0;
			if ((keypoints_1[k1].pt.y < (keypoints_2[k2].pt.y + 5))
			&& (keypoints_1[k1].pt.y > (keypoints_2[k2].pt.y - 5)))
			{
				for (int col = 0;col < descriptors_1.cols;++col)
				{
					double diff = ((float)descriptors_1.at<float>(k1, col) - (float)descriptors_2.at<float>(k2, col));
					sum = sum + pow(diff, 2);
				}
				if (sum < minSSD)
				{
					minSSD = sum;
					matchpairs[k1].queryIdx = k1;
					matchpairs[k1].trainIdx = k2;
					matchpairs[k1].distance = abs((float)keypoints_1[k1].pt.x - (float)keypoints_2[k2].pt.x);
				}
			}
		}
	}

	double max_dist = 0; double min_dist = 100;
	//-- Calculate of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matchpairs[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);
	
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist)
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matchpairs[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matchpairs[i]);
		}
	}
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imshow("Good Matches", img_matches);
	imwrite("SIFT_GoodMatches.png", img_matches);
	
	float disp = 0; //Disparity
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		disp = disp + good_matches[i].distance;
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

	disp = disp / (int)good_matches.size();
	printf("-- Disparity : %f \n", disp);
	int d = (int)disp;
	int maxint = 0;

	for (int i = 0; i < DepthImg.rows; i++)
	{
		for (int j =0; j < DepthImg.cols; ++j)
		{
			int xvalue = max(j - d, 0);
			int v1 = (int)img_1.at<uchar>(i, j);
			int v2 = (int)img_2.at<uchar>(i, xvalue);
			DepthImg.at<uchar>(i, j) = (uchar)(int)abs(v1-v2);
			if ((int)abs(v1 - v2) > maxint)
				maxint = (int)abs(v1 - v2);
		}
	}

	//Normalize max intensity to 255
	float adjustintensity = (float) 255.0 / maxint;
	
	for (int i = 0; i < DepthImg.rows; i++)
	{
		for (int j = 0; j < DepthImg.cols; ++j)
		{
			int k = (int)DepthImg.at<uchar>(i, j);
			DepthImg.at<uchar>(i, j) = (uchar)((int)(k*adjustintensity));
		}
	}

	//Calculate RMSE
	double RMSE = 0;

	for (int i = 0; i < (int)good_matches.size(); ++i)
	{
		int v1 = (int)DepthImg.at<uchar>((int)keypoints_1[good_matches[i].queryIdx].pt.y, (int)keypoints_1[good_matches[i].queryIdx].pt.x);
		int v2 = (int)gtimg.at<uchar>((int)keypoints_1[good_matches[i].queryIdx].pt.y, (int)keypoints_1[good_matches[i].queryIdx].pt.x);
		
		RMSE = RMSE + pow(abs(v1 - v2), 2);
	}

	float RMSEMean = (float)(RMSE / (int)keypoints_1.size());
	RMSE = (double)sqrtf(RMSEMean);

	cout << "Rmse " << RMSE << endl;
	
	imshow("Depth", DepthImg);
	imwrite("SIFT_Depth.png", DepthImg);
	waitKey(0);
	return 0;
}
