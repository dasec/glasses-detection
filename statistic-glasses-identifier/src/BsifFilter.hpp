#ifndef BSIFFILTER_HPP
#define BSIFFILTER_HPP

#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <fstream>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <stddef.h>
#include <matio.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <time.h>

#include "Filter.hpp"
#include "bsif.h"

class BsifFilter : public Filter {
public:

    BsifFilter(const std::string& filterPath) {
        int reading_code = read_filter(filter_dims, &filter_data, filterPath.c_str(), matrix_name);
        CHECK(reading_code == 0) << "Filter \"" << filterPath << "\" could not be loaded";
        bsif_initialize();
    }

    virtual ~BsifFilter() {
        bsif_terminate();
        delete filter_data;
    }

    void createHistogram(double** histogram, size_t& length, cv::Mat img) override {
        featureExtract(histogram, length, img, filter_data, filter_dims);
    }

private:

    const char* matrix_name = "ICAtextureFilters";
    int filter_dims[3];
    double *filter_data;

};

#endif /* BSIFFILTER_HPP */

