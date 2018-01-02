#ifndef EVALUATIONALGORITHM_HPP
#define EVALUATIONALGORITHM_HPP

#define REFLECTION_BLOCK_SIZE 30
#define REFLECTION_BLOCK_STRIDE 10
#define RELATIVE_THRESHOLD 0.97
#define DILATATION_SIZE 3

#include <iostream>
#include <fstream>

#include "ImageManipulation.hpp"
#include "PathFinding.hpp"

unsigned int calculateAreaScore(const cv::Mat& mat, unsigned int rowStart, unsigned int rowEnd, unsigned int colStart, unsigned int colEnd) {

    unsigned int sum = 0;
    for (unsigned int r = rowStart; r < rowEnd; r++) {
        for (unsigned int c = colStart; c < colEnd; c++) {
            sum += mat.at<uchar>(r, c);
        }
    }

    return sum / (double) ((colEnd - colStart)*(rowEnd - rowStart));
}

double evaluateImageReflection(const cv::Mat& mat) {
    double averageBrightness = cv::sum(mat)[0] / (mat.rows * mat.cols);
    double highestReflection = 0;
    int halfBlockSize = REFLECTION_BLOCK_SIZE / 2;

#ifdef DEBUG
    std::ofstream brightnessFile;
    brightnessFile.open("debug_brightness_matrix.csv", std::ios::out);
#endif

    for (int row = halfBlockSize; row < mat.rows - halfBlockSize; row += REFLECTION_BLOCK_STRIDE) {

        double localHighestReflection = 0;

        for (int col = halfBlockSize; col < mat.cols - halfBlockSize; col += REFLECTION_BLOCK_STRIDE) {
            unsigned int averageScore = calculateAreaScore(mat, row - halfBlockSize, row + halfBlockSize, col - halfBlockSize, col + halfBlockSize);
            double brightness = averageScore / averageBrightness;
#ifdef DEBUG
            brightnessFile << col << ";" << mat.rows - row << ";" << brightness << std::endl;
#endif  
            if (localHighestReflection < brightness) localHighestReflection = brightness;

        }

        if (localHighestReflection > highestReflection) highestReflection = localHighestReflection;

#ifdef DEBUG
        brightnessFile << std::endl;
#endif  
    }

#ifdef DEBUG
    brightnessFile.close();
#endif

    DLOG(INFO) << "Highest reflection score: " << highestReflection;
    return highestReflection;
}

double evaluateImageHorizontalBorder(const cv::Mat& mat) {
    cv::Mat matrix(3, 3, CV_32FC1, cv::Scalar(0.0));
    matrix.at<float>(1, 1) = -1;
    matrix.at<float>(2, 1) = 1;

    cv::Mat matMatrix;
    matrixManipulation(matMatrix, mat, matrix);
    matrix.release();

    cv::Mat matThreshold;
    doRelativThreshold(matThreshold, matMatrix, RELATIVE_THRESHOLD);
    matMatrix.release();

    cv::Mat matDilation;
    doDilation(matDilation, matThreshold, DILATATION_SIZE);

    float ratio = findPathSize(matDilation);
    matDilation.release();

    return ratio;
}

#endif /* EVALUATIONALGORITHM_HPP */

