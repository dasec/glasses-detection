#include <caffe/caffe.hpp>
#include <caffe/net.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include "boost/shared_ptr.hpp"
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/core/types_c.h>
#include <glog/logging.h>
#include <list>
#include <glog/logging.h>

#define IMG_WIDTH 320
#define IMG_HEIGHT 240
#define CATEGORIES 2

using namespace boost::filesystem;

float threshold = 0;

std::string getInputParameter(int argc, const char** argv, const char* param) {
    for (int i = 0; i < argc - 1; i++) {
        if (strcmp(argv[i], param) == 0) {
            return std::string(argv[i + 1]);
        }
    }
    return std::string();
}

bool existsParameter(int argc, const char** argv, const char* param) {
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], param) == 0) {
            return true;
        }
    }
    return false;
}

caffe::Net<float>* createCaffeNet(int argc, const char** argv, const std::string& modelFile, const std::string& trainingFile) {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    
    DLOG(INFO) << "Init network";

    caffe::Net<float>* net = new caffe::Net<float>(modelFile, caffe::TEST, 0);
    net->CopyTrainedLayersFrom(trainingFile);

    std::vector<int> shape;
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(IMG_HEIGHT);
    shape.push_back(IMG_WIDTH);

    net->input_blobs()[0]->Reshape(shape);
    net->Reshape();
    return net;
}

void checkNet(const caffe::Net<float>* net) {
    DLOG(INFO) << "Check net";
    
    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";

    CHECK_EQ(net->input_blobs()[0]->channels(), 1) << "Input layer should have 1 channel.";
    CHECK_EQ(net->output_blobs()[0]->channels(), CATEGORIES) << "Output layer should have 2 channel.";

    CHECK(net->input_blobs()[0]->width() == IMG_WIDTH && net->input_blobs()[0]->height() == IMG_HEIGHT) << "Input layer should have the dimension 640x480";
    CHECK(net->output_blobs()[0]->width() == 1 && net->output_blobs()[0]->height() == CATEGORIES) << "Output layer should have the dimension 1x2";
}

void checkImage(const std::string& imgPath, const cv::Mat& image) {
    DVLOG(1) << "Check image";
    
    CHECK(!image.empty()) << "Unable to decode input image " << imgPath;

    CHECK(image.channels() == 1) << "Image " << imgPath << " has to be grayscale";
    CHECK(image.cols == IMG_WIDTH) << "Width of img " << imgPath << " has to be " << IMG_WIDTH;
    CHECK(image.rows == IMG_HEIGHT) << "Height of img " << imgPath << " has to be " << IMG_HEIGHT;
}

void writeImageInNet(caffe::Net<float>* net, const cv::Mat& img) {
    DVLOG(1) << "Write image into network";
    
    cv::Mat channel(img.rows, img.cols, CV_32FC1, (void*) (net->input_blobs()[0]->mutable_cpu_data()));

    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            channel.at<float>(row, col) = img.at<uchar>(row, col) / 255.0;
        }
    }
}

void calcScore(caffe::Net<float>* net, const cv::Mat& img, float& f1, float& f2) {
    DVLOG(1) << "Calculate score";
    writeImageInNet(net, img);

    float loss = 0;
    const std::vector<caffe::Blob<float>*>& top = net->ForwardPrefilled(&loss);

    const float *catScore = top[0]->cpu_data();
    f1 = catScore[0];
    f2 = catScore[1];
}

void readListFile(std::list<std::string>& pathList, const std::string& filepath) {
    std::ifstream in(filepath);
    std::string line;
    while (std::getline(in, line)) pathList.push_back(line);
}

void showImage(const cv::Mat& image) {
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Display window", image); // Show our image inside it.

    cv::waitKey(0);
}

bool isLeft(float scoreA, float scoreB) {
    if(scoreA < threshold) return true;
    else return false;
}

void printResult(const std::string& imgPath, float scoreA, float scoreB, float threshold, const std::string& mode) {
    std::cout << imgPath << ';';
    if (mode == "scoring") {
        std::cout << scoreA << ';' << scoreB << std::endl;
    } else {
        if (isLeft(scoreA, scoreB)) std::cout << "glasses" << std::endl;
        else std::cout << "no_glasses" << std::endl;
    }
}

int main(int argc, const char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_v = 0;
    DLOG(INFO) << "Starting program";
    //--check parameter syntax----------------------------------------

    std::string modelFile = getInputParameter(argc, argv, "--model");
    std::string trainedFile = getInputParameter(argc, argv, "--train");
    std::string inputFile = getInputParameter(argc, argv, "--input");
    std::string rootDirectory = getInputParameter(argc, argv, "--root");
    bool inputIsList = existsParameter(argc, argv, "--list");
    threshold = atof(getInputParameter(argc, argv, "--threshold").c_str());
    std::string mode = getInputParameter(argc, argv, "--mode");
    //bool verbose = existsParameter(argc, argv, "--verbose");

    if (modelFile.empty() || trainedFile.empty() || inputFile.empty() || (inputIsList && rootDirectory.empty()) || mode.empty()) {
        std::cout << "Usage: [PROGRAM] [PARAMETER]" << std::endl;
        std::cout << "Parameter:" << std::endl;
        std::cout << "--model\t\tPath to the model file" << std::endl;
        std::cout << "--train\t\tPath to the training file" << std::endl;
        std::cout << "--input\t\tPath to the histogram file" << std::endl;
        std::cout << "--list\t\t[OPT] Path to a path list file" << std::endl;
        std::cout << "--root\t\t[OPT] Root directory for the list paths" << std::endl;
        std::cout << "--threshold\tThreshold for categorisation" << std::endl;
        std::cout << "--mode\t\tscoring or categorisation" << std::endl;
        return 1;
    }

    //--check validation--------------------------------------------------
    path pathModelFile(modelFile);
    CHECK(exists(pathModelFile)) << "Model file \"" << modelFile << "\" does not exists.";

    path pathTrainedFile(trainedFile);
    CHECK(exists(pathTrainedFile)) << "Trained file \"" << trainedFile << "\" does not exists.";

    path pathInputFile(inputFile);
    CHECK(exists(pathInputFile)) << "Input file \"" << inputFile << "\" does not exists.";

    CHECK(mode == "scoring" || mode == "categorisation") << "Mode parameter has to be \"scoring\" or \"categorisation\"";

    if(mode == "categorisation") CHECK(threshold > 0.0) << "If categorisation mode is on you have to set the threshold parameter";

    if (rootDirectory.empty()) rootDirectory = "./";

    //--create image------------------------------------------------------

    caffe::Net<float>* net = createCaffeNet(argc, argv, modelFile, trainedFile);

    //--classifie image(s)-------------------------------------------------

    if (!inputIsList) {
        DVLOG(1) << "Read image \"" << inputFile << "\"";
        cv::Mat inputImg = cv::imread(inputFile, CV_LOAD_IMAGE_GRAYSCALE);
        if (inputImg.cols != IMG_WIDTH || inputImg.rows != IMG_HEIGHT) {
            DVLOG(1) << "Scale image \"" << inputFile << "\"";
            cv::Mat scaledMat;
            cv::resize(inputImg, scaledMat, cv::Size(IMG_WIDTH, IMG_HEIGHT), 0, 0, CV_INTER_CUBIC);
            inputImg = scaledMat;
        }
        checkImage(inputFile, inputImg);
        float scoreA, scoreB;
        calcScore(net, inputImg, scoreA, scoreB);
        printResult(inputFile, scoreA, scoreB, threshold, mode);
    } else {
        std::list<std::string> pathList;
        readListFile(pathList, inputFile);

        for (std::list<std::string>::const_iterator it = pathList.cbegin(); it != pathList.cend(); it++) {
            DVLOG(1) << "Read image \"" << rootDirectory + *it << "\"";
            cv::Mat inputImg = cv::imread(rootDirectory + *it, CV_LOAD_IMAGE_GRAYSCALE);
            if (inputImg.cols != IMG_WIDTH || inputImg.rows != IMG_HEIGHT) {
                DVLOG(1) << "Scale image \"" << inputFile << "\"";
                cv::Mat scaledMat;
                cv::resize(inputImg, scaledMat, cv::Size(IMG_WIDTH, IMG_HEIGHT), 0, 0, CV_INTER_CUBIC);
                inputImg = scaledMat;
            }
            checkImage(*it, inputImg);
            float scoreA, scoreB;
            calcScore(net, inputImg, scoreA, scoreB);
            printResult(*it, scoreA, scoreB, threshold, mode);
        }

    }
    return 0;
}
