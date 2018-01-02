#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/core.hpp>
#include <glog/logging.h>

class Filter {
public:
    virtual ~Filter() {

    }

        /**
     * Transforms a part of a texture to a histogram
     * @param histogram     the destination of the histogram
     * @param cell          the part of the texture
     * @param sourceTexture the source texture
     * @param shift         the alignment of pixels for shifting
     */
    void toHistogram(uint16_t *histogram, const cv::Mat& sourceTexture) {
        DVLOG(1) << "Create histogram from mat";

        for (int i = 0; i < 256; i++) histogram[i] = 0;

        for (int row = 0; row < sourceTexture.rows; row++) {
            for (int col = 0; col < sourceTexture.cols; col++) {
                histogram[*sourceTexture.ptr<uchar>(row, col)]++;
            }
        }
    }

    void toNormalizedHistogram(double* normHistogram, uint16_t* histogram, size_t length) {
        DVLOG(1) << "Normalize histogram";
        
        size_t sum = 0;
        for (unsigned int i = 0; i < length; i++) sum += histogram[i];

        for (unsigned int i = 0; i < length; i++) {
            normHistogram[i] = histogram[i] / ((double) (sum));
        }
    }

    virtual void createHistogram(double** histogram, size_t& length, cv::Mat img) = 0;
};

#endif /* FILTER_HPP */

