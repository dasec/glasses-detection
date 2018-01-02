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

#include "EvaluationAlgorithm.hpp"
#include "ParameterHandler.hpp"

#define REFLECTION_THRESHOLD 2.0
#define FRAME_THRESHOLD 4.5

enum Mode {
    SCORING, CATEGORISATION
};

void printHelp(const char* startCommand) {
    std::cout << "Syntax: " << startCommand << std::endl;
    std::cout << "-i/--input\t" << "Input directory which contains the images" << std::endl;
    std::cout << "-f/--format\t" << "[OPTIONAL (default: all)] The file extension of the images which should be analysed" << std::endl;
    std::cout << "-m/--mode\t" << "[OPTIONAL (default: CATEGORISATION)] The output mode (SCORING or CATEGORISATION)" << std::endl;
}

size_t countFilesInDir(const boost::filesystem::path& dir, const std::string& fileExtension) {
    size_t counter = 0;
    for (boost::filesystem::recursive_directory_iterator it(dir); it != boost::filesystem::recursive_directory_iterator(); ++it) {
        //if (fileExtension != it->path().extension().string()) continue; //feature disabled
        counter++;
    }
    return counter;
}

void evaluateImages(const std::string& inputDir, const std::string& fExt, Mode mode) {
    DLOG(INFO) << "Evaluate directory " << inputDir;
    
    boost::filesystem::path inputPath(inputDir);
    CHECK(boost::filesystem::exists(inputPath)) << "Input directory does not exists";

    size_t numberInputFiles = countFilesInDir(inputPath, fExt);

    std::cerr.precision(4);

    int counter = 1;
    boost::filesystem::recursive_directory_iterator end;
    for (boost::filesystem::recursive_directory_iterator dir(inputPath); dir != end; dir++) {
        boost::filesystem::path filePath = dir->path();
        std::string fileExtension = filePath.extension().string();

        /*if (fileExtension.empty() == false){ //feature disabled
            if(fileExtension != fExt) continue;
        }*/

        DLOG(INFO) << "Evaluate file " << filePath.string();

        cv::Mat mat = cv::imread(filePath.string(), CV_LOAD_IMAGE_GRAYSCALE);
        CHECK(!mat.empty()) << "Image " << filePath.string() << " could not be readed";

        double reflectionScore = evaluateImageReflection(mat);
        double horizontalBorderScore = evaluateImageHorizontalBorder(mat);

        std::cerr << "Progress: " << (counter * 100) / (double) (numberInputFiles) << "\t% [" << counter << "/" << numberInputFiles << "]                                                    \r";
        std::cout << filePath.filename().string() << ";";

        if (mode == CATEGORISATION) {
            if (reflectionScore >= REFLECTION_THRESHOLD || horizontalBorderScore >= FRAME_THRESHOLD) std::cout << "1" << std::endl;
            else std::cout << "0" << std::endl;
        } else {
            std::cout << reflectionScore << ";" << horizontalBorderScore << std::endl;
        }

        counter++;
    }
}

void evaluateImage(const std::string& filePath, Mode mode) {
    DLOG(INFO) << "Evaluate file " << filePath;

    cv::Mat mat = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
    CHECK(!mat.empty()) << "Image " << filePath << " could not be readed";

    double reflectionScore = evaluateImageReflection(mat);
    double horizontalBorderScore = evaluateImageHorizontalBorder(mat);
    
    boost::filesystem::path p(filePath);

    if (mode == CATEGORISATION) {
        if (reflectionScore >= REFLECTION_THRESHOLD || horizontalBorderScore >= FRAME_THRESHOLD) std::cout << p.filename() << ";1" << std::endl;
        else std::cout << p.filename() << ";0" << std::endl;
    } else {
        std::cout << p.filename() << ";" << reflectionScore << ";" << horizontalBorderScore << std::endl;
    }
}

void setStackSize(size_t stackSize) {
    const rlim_t kStackSize = stackSize;
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0) {
        if (rl.rlim_cur < kStackSize) {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0) {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }
        }
    }
}

int main(int argc, const char** argv) {

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 0;

    setStackSize(64 * 1024 * 1024); // stack size = 64 MB

    std::string inputPath = getInputParameter(argc, argv, "-i|--input");
    std::string fileExt = getInputParameter(argc, argv, "-f|--format");

    std::string mode = getInputParameter(argc, argv, "-m|--mode");

    if (inputPath.empty()) {
        printHelp(argv[0]);
        exit(1);
    }

    Mode outputMode = CATEGORISATION;
    if (mode == "SCORING") outputMode = SCORING;

    boost::filesystem::path p(inputPath);
    
    if(boost::filesystem::is_directory(p)) evaluateImages(inputPath, fileExt, outputMode);
    else evaluateImage(inputPath, outputMode);

    return 0;
}
