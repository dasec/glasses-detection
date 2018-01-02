#ifndef LBPFILTER_HPP
#define LBPFILTER_HPP

#define LBP_FILTER_SIZE 15
#define LBP_HISTOGRAM_LENGTH 256

#include "Filter.hpp"

class LbpFilter : public Filter {
public:

    LbpFilter() {

    }

    virtual ~LbpFilter() {

    }

    void createHistogram(double** histogram, size_t& length, cv::Mat img) override {
        DLOG(INFO) << "Calculate lbp statistic score";

        cv::Mat code;
        extractTextureWithLbp(code, LBP_FILTER_SIZE, img);

#ifdef DEBUG
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
        cv::imshow("Display window", code); // Show our image inside it.
        cv::waitKey(0);
#endif

        uint16_t unHistogram[LBP_HISTOGRAM_LENGTH];
        toHistogram(unHistogram, code);

        length = LBP_HISTOGRAM_LENGTH;
        *histogram = new double[length];
        toNormalizedHistogram(*histogram, unHistogram, LBP_HISTOGRAM_LENGTH);
    }

private:

    struct Cell {
        int startX, startY;
        int endX, endY;
        int width, height;
    };

    uchar getAverageValue(const int centerX, const int centerY, const Cell& cell, const int filterSize, const cv::Mat& texture) {
        int averageValue = 0;
        int mod_x, mod_y;
        const int halfCellSize = filterSize / 2;

        for (int y = centerY - halfCellSize; y <= centerY + halfCellSize; y++) {

            mod_y = y;
            if (mod_y < cell.startY) mod_y += cell.height;
            else if (mod_y >= cell.endY) mod_y -= cell.height;

            for (int x = centerX - halfCellSize; x <= centerX + halfCellSize; x++) {

                mod_x = x;
                if (mod_x < cell.startX) mod_x += cell.width;
                else if (mod_x >= cell.endX) mod_x -= cell.width;

                averageValue += *texture.ptr<uchar>(mod_y, mod_x);
            }
        }

        return (uchar) (averageValue / pow(filterSize, 2));
    }

    /**
     * Calculate the LBP-Value clockwise of a pixel
     * @param x             the x-coordinate of the pixel in the texture
     * @param y             the y-coordinate of the pixel in the texture
     * @param filterSize    the filter-size of the MB-LBP algorithm (has to be a multiple of 3)
     * @param texture       the iris texture
     * @return              the clockwise LBP-Value of this pixel
     */
    uchar calculateLBPValue(const int centerY, const int centerX, const Cell& cellSize, const int filterSize, const cv::Mat& texture) {
        DVLOG(3) << "Calculate LBP value for pixel " << centerX << ":" << centerY;

        const int filterCellSize = filterSize / 3;
        const uchar centerValue = getAverageValue(centerX, centerY, cellSize, filterCellSize, texture);

        uchar LBPValue = 0x00;
        const uchar bitMask = 0x01;

        //----------------------Top-Neighbours--------------------------------------    
        for (int x = centerX - filterCellSize; x <= centerX + filterCellSize; x += filterCellSize) {
            if (getAverageValue(x, centerY - filterCellSize, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
            LBPValue = LBPValue << 1; //*2
        }
        //----------------------Right-Neighbour------------------------------------
        if (getAverageValue(centerX + filterCellSize, centerY, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
        LBPValue = LBPValue << 1; //*2
        //----------------------Bottom-Neighbours----------------------------------
        for (int x = centerX + filterCellSize; x >= centerX - filterCellSize; x -= filterCellSize) {
            if (getAverageValue(x, centerY + filterCellSize, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;
            LBPValue = LBPValue << 1; //*2
        }
        //-----------------------Left-Neighbours-----------------------------------
        if (getAverageValue(centerX - filterCellSize, centerY, cellSize, filterCellSize, texture) >= centerValue) LBPValue |= bitMask;

        return LBPValue;
    }

    /**
     * Extracts the features of the complete iris texture with the lbp algorithm
     * @param extractedData     the destination for the extracted data
     * @param cell              the start- and end-positions of this cell in the texture
     * @param texture           the source iris-texture
     */
    void extractTextureWithLbp(cv::Mat& extractedData, const int filterSize, const cv::Mat& texture) {
        DVLOG(1) << "Extract lbp texture";

        Cell cell;
        cell.startX = 0;
        cell.startY = 0;
        cell.width = texture.cols;
        cell.height = texture.rows;
        cell.endX = cell.startX + cell.width;
        cell.endY = cell.startY + cell.height;

        extractedData = cv::Mat(texture.rows, texture.cols, CV_8UC1);

#pragma omp parallel for
        for (int row = 0; row < texture.rows; row++) {
            for (int col = 0; col < texture.cols; col++) {
                *extractedData.ptr<uchar>(row, col) = calculateLBPValue(row, col, cell, filterSize, texture);
            }
        }
    }

};

#endif /* LBPFILTER_HPP */

