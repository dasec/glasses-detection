#include <iostream>
#include <string>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>
#include <sys/resource.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/lambda/bind.hpp>
#include <vector>

#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "ParameterHandler.hpp"
#include "Filter.hpp"
#include "LbpFilter.hpp"
#include "BsifFilter.hpp"

#define LBP_FILTER_SIZE 15
#define LBP_HISTOGRAM_LENGTH 256

Filter* filter = nullptr;

enum Status {
    RUNNING, STOPPED
};

enum InputType{
    SINGLE, MULTI
};

enum Mode {
    SCORING, CATEGORISATION, HISTOGRAM
};

Status status = RUNNING;
InputType inputType = SINGLE;

void handler(int signum){ 
    status = STOPPED;
    if(inputType == SINGLE) exit(1);
} 

void printHelp(const char* startCommand) {
    std::cout << "Syntax: " << startCommand << std::endl;
    std::cout << "-i/--input\t" << "Input directory which contains the images" << std::endl;
    std::cout << "-a/--alg\t" << "Algorithm for processing: lbp or bsif" << std::endl;
    std::cout << "-f/--filter\t" << "[IF alg=bsif] File of the bsif filter which should be used" << std::endl;
    std::cout << "-m/--mode\t" << "[OPTIONAL (default: CATEGORISATION)] The output mode (SCORING or CATEGORISATION)" << std::endl;
}

size_t countFilesInDir(const boost::filesystem::path& dir) {
    size_t counter = 0;
    for (boost::filesystem::recursive_directory_iterator it(dir); it != boost::filesystem::recursive_directory_iterator(); ++it) {
        //if (fileExtension != it->path().extension().string()) continue; //feature disabled
        counter++;
    }
    return counter;
}

void printHistogram(double* histogram, size_t length) {
    DVLOG(1) << "Print histogram";
    for (unsigned int i = 0; i < length; i++) {
        std::cout << i << ";" << histogram[i] << std::endl;
    }
}

void calculateHistogramCharateristics(std::vector<double*>& histograms, size_t histogramLength){
    DVLOG(1) << "Calculate average histogram";
    double averageHistogram[histogramLength];
    for(unsigned int vIndex=0; vIndex<histogramLength; vIndex++){
        averageHistogram[vIndex] = 0;
        for(unsigned int histIndex=0; histIndex<histograms.size(); histIndex++){
            averageHistogram[vIndex] += histograms[histIndex][vIndex];
        }
    }
    for(unsigned int vIndex=0; vIndex<histogramLength; vIndex++) averageHistogram[vIndex] /= (double)  (histograms.size());
    
    DVLOG(1) << "Calculate median histogram";
    double medianHistogram[histogramLength];
    for(unsigned int vIndex=0; vIndex<histogramLength; vIndex++){
        std::vector<double> values;
        for(unsigned int histIndex=0; histIndex<histograms.size(); histIndex++) values.push_back(histograms[histIndex][vIndex]);
        std::sort(values.begin(), values.end());
        medianHistogram[vIndex] = values[values.size()/2];
    }
    
    DVLOG(1) << "Calculate standard deviation histogram";
    double standardDeviations[histogramLength];    
    for(unsigned int valueIndex=0; valueIndex<histogramLength; valueIndex++){
        double variance = 0;
        for(unsigned int histIndex=0; histIndex<histograms.size(); histIndex++) variance += pow(histograms[histIndex][valueIndex] - medianHistogram[valueIndex], 2);
        standardDeviations[valueIndex] = sqrt(variance/histograms.size());
    }
    
    for(unsigned int i=0; i<histogramLength; i++){
        std::cout << i << ";" << averageHistogram[i] << ";" << medianHistogram[i] << ";" << standardDeviations[i] << std::endl;
    }
    
}

void evaluateImages(const std::string& inputDir, Mode mode) {
    DLOG(INFO) << "Evaluate directory " << inputDir;

    boost::filesystem::path inputPath(inputDir);
    CHECK(boost::filesystem::exists(inputPath)) << "Input directory does not exists";

    size_t numberInputFiles = countFilesInDir(inputPath);

    size_t histogramLength = 0;
    std::vector<double*> histograms;

    std::cerr.precision(4);

    int counter = 1;
    boost::filesystem::recursive_directory_iterator end;
    for (boost::filesystem::recursive_directory_iterator dir(inputPath); dir != end && status == RUNNING; dir++) {
        boost::filesystem::path filePath = dir->path();

        DLOG(INFO) << "Evaluate file " << filePath.string();

        cv::Mat mat = cv::imread(filePath.string(), CV_LOAD_IMAGE_GRAYSCALE);
        CHECK(!mat.empty()) << "Image " << filePath.string() << " could not be readed";

        std::cerr << "Progress: " << (counter * 100) / (double) (numberInputFiles) << "\t% [" << counter << "/" << numberInputFiles << "]                                                    \r";

        double* histogram;
        filter->createHistogram(&histogram, histogramLength, mat);

        histograms.push_back(histogram);

        counter++;
    }

    if (mode == HISTOGRAM){
        calculateHistogramCharateristics(histograms, histogramLength);        
        for(double* hist : histograms) delete[] hist;
    }
}

void evaluateImage(const std::string& filePath, const std::string& bsifFilter, Mode mode) {
    DLOG(INFO) << "Evaluate file " << filePath;

    cv::Mat mat = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
    CHECK(!mat.empty()) << "Image " << filePath << " could not be readed";

    boost::filesystem::path p(filePath);

    double* histogram = nullptr;
    size_t histogramLength = 0;
    filter->createHistogram(&histogram, histogramLength, mat);

    switch (mode) {
        case HISTOGRAM:
            printHistogram(histogram, histogramLength);
            break;
        case CATEGORISATION:
            break;
        case SCORING:
            break;
    }

    delete histogram;
}

int main(int argc, const char** argv) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 2;
    
    signal(SIGINT, handler);

    std::string inputPath = getInputParameter(argc, argv, "-i|--input");
    std::string bsifFilter = getInputParameter(argc, argv, "-f|--filter");

    std::string alg = getInputParameter(argc, argv, "-a|--alg");

    std::string mode = getInputParameter(argc, argv, "-m|--mode");

    if (inputPath.empty()) {
        printHelp(argv[0]);
        exit(1);
    }

    CHECK(alg == "lbp" || alg == "bsif") << "Algorithm has to be lbp or bsif";

    if (alg == "bsif") {
        CHECK(!bsifFilter.empty()) << "When using bsif, the filter parameter has to be setted";
        filter = new BsifFilter(bsifFilter);
    } else {
        filter = new LbpFilter();
    }

    Mode outputMode = CATEGORISATION;
    if (mode == "SCORING" || mode == "scoring") outputMode = SCORING;
    else if (mode == "HISTOGRAM" || mode == "HIST" || mode == "hist" || mode == "histogram") outputMode = HISTOGRAM;

    boost::filesystem::path p(inputPath);

    if (boost::filesystem::is_directory(p)){
        inputType = MULTI;
        evaluateImages(inputPath, outputMode);
    }
    else{
        inputType = SINGLE;
        evaluateImage(inputPath, bsifFilter, outputMode);
    }

    return 0;
}
