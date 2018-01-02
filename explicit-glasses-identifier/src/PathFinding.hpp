#ifndef PATHFINDING_HPP
#define PATHFINDING_HPP

#include <deque>
#include <unordered_map>
#include <map>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>

#define MIN_HORIZONTAL_LENGTH 120
#define BORDER_SPACE_ROW 0.10
#define BORDER_SPACE_COL 0.10
#define BORDER_STRIDE_ROW 5
#define BORDER_STRIDE_COL 5

unsigned int getAreaSize(const cv::Mat& mat, int cat){
    unsigned int counter = 0;
    for(int row=0; row < mat.rows; row++){
        for(int col=0; col < mat.cols; col++){
            if(mat.at<ushort>(row, col) == cat) counter++;
        }
    }
    return counter;
}

void catPoints(const cv::Mat& srcMat, cv::Mat& catMat, int midPointRow, int midPointCol, ushort cat, int& lowestCol, int& highestCol, int& lowestRow, int& highestRow){
        catMat.at<ushort>(midPointRow, midPointCol) = cat;
        
        if(midPointCol < lowestCol) lowestCol = midPointCol;
        if(midPointCol > highestCol) highestCol = midPointCol;
        if(midPointRow < lowestRow) lowestRow = midPointRow;
        if(midPointRow > highestRow) highestRow = midPointRow;
       
        for(int row = midPointRow-1; row <= midPointRow+1; row++){
                for(int col = midPointCol-1; col <= midPointCol+1; col++){
                                            
                        if(row < 0 || row >= srcMat.rows) continue;
                        if(col < 0 || col >= srcMat.cols) continue;
                        
                        if(srcMat.at<uchar>(row, col) < 128) continue;
                        
                        if(catMat.at<ushort>(row, col) <= 0){
                                catPoints(srcMat, catMat, row, col, cat, lowestCol, highestCol, lowestRow, highestRow);
                        }        
                }
        }
}

void findStartPoints(const cv::Mat& mat, std::deque<cv::Point2i>& points, int colSpace, int rowSpace, float borderPlaceRow, float borderPlaceCol){        
        for(int row = 0; row < mat.rows; row+=rowSpace){                        //left
                for(int col=0; col < mat.cols*borderPlaceCol; col+=colSpace){
                        if(mat.at<uchar>(row, col) > 128){
                                points.push_back(cv::Point2i(col, row));
                                DVLOG(1) << "Find start point row: " << row << " col: " << col;
                        }
                }
        }
        
        for(int row = 0; row < mat.rows; row+=rowSpace){                        //right
                for(int col=mat.cols*(1-borderPlaceCol); col < mat.cols; col+=colSpace){
                        if(mat.at<uchar>(row, col) > 128){
                                points.push_back(cv::Point2i(col, row));
                                DVLOG(1) << "Find start point row: " << row << " col: " << col;
                        }
                }
        }
        
       /* for(int row = 0; row < mat.rows*borderPlaceRow; row+=rowSpace){                      //top
                for(int col=mat.cols*borderPlaceCol; col < mat.cols*(1-borderPlaceCol); col+=colSpace){
                        if(mat.at<uchar>(row, col) > 128){
                                points.push_back(cv::Point2i(col, row));
                                //DLOG(INFO) << "Find start point row: " << row << " col: " << col;
                        }
                }
        }*/
        
        for(int row = mat.rows*(1-borderPlaceRow); row < mat.rows; row+=rowSpace){     //bottom between the left/right areas
                for(int col=mat.cols*borderPlaceCol; col < mat.cols*(1-borderPlaceCol); col+=colSpace){
                        if(mat.at<uchar>(row, col) > 128){
                                points.push_back(cv::Point2i(col, row));
                                DVLOG(1) << "Find start point row: " << row << " col: " << col;
                        }
                }
        }
}

float findPathDimension(cv::Mat& matSrc){
        std::deque<cv::Point2i> points;
        findStartPoints(matSrc, points, BORDER_STRIDE_COL, BORDER_STRIDE_ROW, BORDER_SPACE_COL, BORDER_SPACE_ROW);
        
        int newCat = 65000;
        cv::Mat matCat(matSrc.rows, matSrc.cols, CV_16UC1, cv::Scalar(0));
        
        float maxRatio = 0;
        
        for(auto it = points.begin(); it != points.end(); it++){
                if(matCat.at<ushort>(it->y, it->x) <= 0){
                        
                        int lowestCol = it->x;
                        int highestCol = it->x;
                        int  lowestRow = it->y;
                        int highestRow = it->y;
                        
                        catPoints(matSrc, matCat, it->y, it->x, newCat++, lowestCol, highestCol, lowestRow, highestRow);
                        
                        double ratio = (highestCol-lowestCol)/(double)(highestRow-lowestRow);
                        if(ratio > maxRatio && highestCol-lowestCol >= MIN_HORIZONTAL_LENGTH) maxRatio = ratio;
                        
                        int width = highestCol-lowestCol;
                        int height = highestRow-lowestRow;
                        DLOG(INFO) << "Cat " << newCat-1 << " Width: " << width << " Height: " <<  height << " Ratio: " << width/(double)(height);
                }                   
        }
       
        return maxRatio;
}

float findPathSize(cv::Mat& matSrc){
        std::deque<cv::Point2i> points;
        findStartPoints(matSrc, points, BORDER_STRIDE_COL, BORDER_STRIDE_ROW, BORDER_SPACE_COL, BORDER_SPACE_ROW);
        
        int catValue = 65000;
        cv::Mat matCat(matSrc.rows, matSrc.cols, CV_16UC1, cv::Scalar(0));
        
        float maxRatio = 0;
        
        for(auto it = points.begin(); it != points.end(); it++){
                if(matCat.at<ushort>(it->y, it->x) <= 0){
                        
                        int lowestCol = it->x;
                        int highestCol = it->x;
                        int  lowestRow = it->y;
                        int highestRow = it->y;
                        
                        catPoints(matSrc, matCat, it->y, it->x, catValue, lowestCol, highestCol, lowestRow, highestRow);

                        double ratioDimension = (highestCol-lowestCol)/(double)(highestRow-lowestRow);
                        if(ratioDimension > maxRatio && highestCol-lowestCol >= MIN_HORIZONTAL_LENGTH) maxRatio = ratioDimension;
                        
                        int width = highestCol-lowestCol;
                        int height = highestRow-lowestRow;
                        DLOG(INFO) << "Cat " << catValue << " Width: " << width << " Height: " <<  height << " Ratio: " << width/(double)(height);
                        
                        catValue++;
                }                   
        }
        
        return maxRatio;
}


#endif /* PATHFINDING_HPP */
