#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

int GroundTruth(Mat& _originalImage)
{

    Mat originalImage(_originalImage.rows + 2, _originalImage.cols + 2, _originalImage.type());
    copyMakeBorder(_originalImage, originalImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(255, 255, 255));

    Mat bwImage(originalImage.size(), CV_8UC1);

    uchar thresholdValue = 100;
    uchar maxValue = 255;
    uchar middleValue = 192;
    uchar zeroValue = 0;
    Scalar middleScalar(middleValue);
    Scalar zeroScalar(zeroValue);

    static int neigborsCount = 4;
    static int dx[] = {-1,  0, 0, 1};
    static int dy[] = { 0, -1, 1, 0};
    int di, rx, ry;
    int perimeter;

    threshold(originalImage, bwImage, thresholdValue, maxValue, THRESH_BINARY_INV);

    int regionsCount = 0;
    int totalPixelCount = bwImage.rows * bwImage.cols;
    Point seedPoint;
    Rect rectFilled;
    int valuesSum, q1, q2, q3;
    bool p00, p10, p01, p11;

    for(int i = 0; i < totalPixelCount; i++)
    {
        if (bwImage.data[i] == maxValue)
        {
            seedPoint.x = i % bwImage.cols;
            seedPoint.y = i / bwImage.cols;

            if ((seedPoint.x == 0) || (seedPoint.y == 0) || (seedPoint.x == bwImage.cols - 1) || (seedPoint.y == bwImage.rows - 1))
            {
                continue;
            }

            regionsCount++;

            size_t pixelsFilled = floodFill(bwImage, seedPoint, middleScalar, &rectFilled);

            perimeter = 0;
            q1 = 0; q2 = 0; q3 = 0;

            int crossings[rectFilled.height];
            for(int j = 0; j < rectFilled.height; j++)
            {
                crossings[j] = 0;
            }

            for(ry = rectFilled.y - 1; ry <= rectFilled.y + rectFilled.height; ry++)
            {
                for(rx = rectFilled.x - 1; rx <= rectFilled.x + rectFilled.width; rx++)
                {
                    if ((bwImage.at<uint8_t>(ry, rx - 1) != bwImage.at<uint8_t>(ry, rx)) && (bwImage.at<uint8_t>(ry, rx - 1) + bwImage.at<uint8_t>(ry, rx) == middleValue + zeroValue))
                    {
                        crossings[ry - rectFilled.y]++;
                    }

                    if (bwImage.at<uint8_t>(ry, rx) == middleValue)
                    {
                        for(di = 0; di < neigborsCount; di++)
                        {
                            int xNew = rx + dx[di];
                            int yNew = ry + dy[di];

                            if (bwImage.at<uint8_t>(yNew, xNew) == zeroValue)
                            {
                                perimeter++;
                            }
                        }
                    }

                    p00 = bwImage.at<uint8_t>(ry, rx) == middleValue;
                    p01 = bwImage.at<uint8_t>(ry, rx + 1) == middleValue;
                    p10 = bwImage.at<uint8_t>(ry + 1, rx) == middleValue;
                    p11 = bwImage.at<uint8_t>(ry + 1, rx + 1) == middleValue;
                    valuesSum = p00 + p01 + p10 + p11;

                    if (valuesSum == 1) q1++; else
                    if (valuesSum == 3) q2++; else
                    if ((valuesSum == 2) && (p00 == p11)) q3++;
                }
            }

            q1 = q1 - q2 + 2 * q3;
            if (q1 % 4 != 0)
            {
                printf("Non-integer Euler number");
                exit(0);
            }
            q1 /= 4;

            /*printf("New region: %d\n", regionsCount);
            printf("Area: %d\n", (int)pixelsFilled);
            printf("Bounding box (%d; %d) + (%d; %d)\n", rectFilled.x - 1, rectFilled.y - 1, rectFilled.width, rectFilled.height);
            printf("Perimeter: %d\n", (int)perimeter);
            printf("Euler number: %d\n", q1);
            printf("Crossings: ");
            for(int j = 0; j < rectFilled.height; j++)
            {
                printf("%d ", crossings[j]);
            }*/

            vector<int> m_crossings;
            m_crossings.push_back(crossings[(int)rectFilled.height/6]);
            m_crossings.push_back(crossings[(int)3*rectFilled.height/6]);
            m_crossings.push_back(crossings[(int)5*rectFilled.height/6]);
            sort(m_crossings.begin(), m_crossings.end());

	   //Features used in the first stage classifier
	   //aspect ratio (w/h), compactness (sqrt(a/p), number of holes (1 − η), and a horizontal crossings feature (cˆ = median {c_1*w/6, c_3*w/6, c_5*w/6}) which estimates number of character strokes in horizontal projection
            if ((rectFilled.width>=20)&&(rectFilled.height>=20)) // TODO find a better way to select good negative examples
		printf("%f,%f,%f,%f\n",(float)rectFilled.width/rectFilled.height, sqrt(pixelsFilled)/perimeter, (float)(1-q1), (float)m_crossings.at(1)); 

            floodFill(bwImage, seedPoint, zeroScalar);

        }
    }

}


int main(int argc, char** argv)
{
    Mat originalImage;

    if (argc == 1)
    {
	exit(0);
    }
    else
    {
    	originalImage = imread(argv[1],0);
      originalImage = 255 - originalImage;
    }


    GroundTruth(originalImage);

    return 0;
}
