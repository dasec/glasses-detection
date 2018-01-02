#ifndef IMAGEMANIPULATION_HPP
#define IMAGEMANIPULATION_HPP

#include <opencv/cv.h>
#include <glog/logging.h>

void matrixManipulation(cv::Mat& matDest, const cv::Mat& matSrc, const cv::Mat& matrix){
        DLOG(INFO) << "Run matrix manipulation";
        
        matDest = matSrc.clone();
        int halfMatrixWidth = matrix.cols/2;
        int halfMatrixHeight = matrix.rows/2;
        
        #pragma omp parallel for
        for(int row = 0; row < matDest.rows; row++){
                for(int col = 0; col < matDest.cols; col++){
                        
                        int value = 0;
                        
                        for(int mRow = 0; mRow < matrix.rows; mRow++){
                                for(int mCol = 0; mCol < matrix.cols; mCol++){
                                        
                                        int rCordinate = row+mRow-halfMatrixHeight;
                                        int cCordinate = col+mCol-halfMatrixWidth;
                                        
                                        if(rCordinate < 0 || matSrc.rows <= rCordinate) continue;
                                        if(cCordinate < 0 || matSrc.cols <= cCordinate) continue;
                                        
                                        value += matSrc.at<uchar>(rCordinate, cCordinate)*matrix.at<float>(mRow,mCol);
                                }
                        }
                        
                        value = value/(matrix.rows*matrix.cols);
                        value += 128;
                        
                        if(value<0) value = 0;
                        if(value>=256) value = 255;
                        
                        matDest.at<uchar>(row, col) = value;                        
                }
        }
}

void doRelativThreshold(cv::Mat& matDest, cv::Mat& matSrc, double relThreshold){
        
        DLOG(INFO) << "Run threshold manipulation";
        
        int histLength = 256;
        unsigned int hist[histLength] = {};
        
        for(int row = 0; row < matSrc.rows; row++){
                for(int col = 0; col < matSrc.cols; col++){
                        hist[matSrc.at<uchar>(row, col)]++;
                }
        }
        
        unsigned int absThreshold = (matSrc.rows*matSrc.cols)*relThreshold;
        
        DLOG(INFO) << "Absolute threshold: " << absThreshold;
        
        unsigned long sum = 0;
        int histIndex = 0;
        for(histIndex = 0; histIndex < histLength; histIndex++){
                sum += hist[histIndex];           
                if(sum > absThreshold) break;
        }
        
        DLOG(INFO) << "Gray threshold: " << histIndex;
        
        cv::threshold(matSrc, matDest, histIndex,histLength-1,CV_THRESH_BINARY);
}

void doErosion(cv::Mat& matDest, const cv::Mat& matSrc, int erosionSize){
        DLOG(INFO) << "Run erosion with erosion size " << erosionSize;
        cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*erosionSize + 1, 2*erosionSize+1 ),
                                       cv::Point( erosionSize, erosionSize ) );
        cv::erode(matSrc, matDest, element );
}

void doDilation(cv::Mat& matDest, const cv::Mat& matSrc, int dilationSize){
        DLOG(INFO) << "Run dilation with dilation size " << dilationSize;
        cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*dilationSize + 1, 2*dilationSize+1 ),
                                       cv::Point( dilationSize, dilationSize ) );
        cv::dilate(matSrc, matDest, element );
}

#endif /* IMAGEMANIPULATION_HPP */

