#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "net.h"
#include "util.h"

int main(int argc, char **argv){
    using namespace std;
    using namespace megEngine;
    const float mean[3] = { 0.485, 0.456, 0.406 };
    const float std[3]  = {0.229, 0.224, 0.225};

    const string engineFile = argv[1];
    const string imgpath    = argv[2];

    net model(engineFile);
    unique_ptr<float []> output(new float[model.mBindBufferSizes[1]]);

    cv::Mat img, show;
    img = cv::imread(imgpath);
    cv::resize(img, img, cv::Size(224, 224));

    // use cpu for image preprocess
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // float *inputDataHost;
    // size_t numInput;
    // numInput = numTensorElements(model.inputDim);
    // inputDataHost = (float*) malloc(numInput * sizeof(float));
    // cvImageToTensor(img, inputDataHost, model.inputDim);
    // preprocessVgg(inputDataHost, model.inputDim, mean, std);

    // warm up
    for(int i = 0;i<10;++i)
        model.infer(img, output.get(), show);

    // test
    auto start = chrono::system_clock::now();
    for(int i = 0;i<100;++i)
    {
        model.infer(img, output.get(), show);
    }
    
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start)/100.0;
    cout << "cost time is:" << double(duration.count()) << "ms" << endl;
    
    return 0;
}