#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "net.h"
#include "util.h"

int main(int argc, char **argv){
    using namespace std;
    using namespace megEngine;

    const string engineFile = argv[1];
    const string imgpath    = argv[2];

    net model(engineFile);
    unique_ptr<float []> output(new float[model.outputDim.d[1]]);

    cv::Mat img;
    img = cv::imread(imgpath);
    cv::resize(img, img, cv::Size(224, 224));

    // use cpu for image preprocess
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // float *inputDataHost;
    // size_t numInput;
    // numInput = numTensorElements(model.inputDim);
    // inputDataHost = (float*) malloc(numInput * sizeof(float));
    // cvImageToTensor(img, inputDataHost, model.inputDim);
    // preprocessVgg(inputDataHost, model.inputDim);

    cv::Mat show;
    auto start = chrono::system_clock::now();
    for(int i = 0;i<100;++i)
    {
        model.infer(img, output.get(), show);
    }
    
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start)/100.0;
    cout << "cost time is:" << double(duration.count()) << "ms" << endl;

    // read labels
    std::vector<size_t> index = argsort(output.get(), model.outputDim);

    cout << "predict: " << index[0] << " " << output[index[0]] << endl;
    
    return 0;
}