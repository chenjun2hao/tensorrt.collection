#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include "net.h"
#include "util.h"

int main(int argc, char **argv){
    using namespace std;
    using namespace megEngine;

    // show color table
    const cv::Vec3b colorMap[]=
    {
        cv::Vec3b(128, 64,128),
        cv::Vec3b(244, 35,232),
        cv::Vec3b( 70, 70, 70),
        cv::Vec3b(102,102,156),
        cv::Vec3b(190,153,153),

        cv::Vec3b(153,153,153),
        cv::Vec3b(250,170, 30),
        cv::Vec3b(220,220,  0),
        cv::Vec3b(107,142, 35),
        cv::Vec3b(152,251,152),

        cv::Vec3b( 70,130,180),
        cv::Vec3b(220, 20, 60),
        cv::Vec3b(255,  0,  0),
        cv::Vec3b(  0,  0,142),
        cv::Vec3b(  0,  0, 70),

        cv::Vec3b(  0, 60,100),
        cv::Vec3b(  0, 80,100),
        cv::Vec3b(  0,  0,230),
        cv::Vec3b(119, 11, 32),
        cv::Vec3b(  0,  0,  0)
    };
    const float mean[3] = {0., 0., 0.};
    const float std[3]  = {1., 1., 1.};

    const string engineFile = argv[1];
    const string imgpath    = argv[2];

    net model(engineFile);
    unique_ptr<int32_t []> output(new int32_t[model.mBindBufferSizes[1]]);

    cv::Mat img;
    img = cv::imread(imgpath);
    cv::resize(img, img, cv::Size(model.inputDim.d[3], model.inputDim.d[2]));

    // use cpu for image preprocess: mat->tensor
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    float *inputDataHost;
    size_t numInput = numTensorElements(model.inputDim);
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    cvImageToTensor(img, inputDataHost, model.inputDim);
    preprocessVgg(inputDataHost, model.inputDim, mean, std);

    cv::Mat show;
    auto start = chrono::system_clock::now();
    for(int i = 0;i<100;++i)
    {
        // preprocess with cuda
        // model.infer(img, output.get(), show);

        // preprocess with cpu
        model.infer(inputDataHost, output.get(), show);

    }
    
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start)/100.0;
    cout << "cost time is:" << double(duration.count()) << "ms" << endl;

    // draw the semantic result
    cv::Mat coloredImg(model.outputDim.d[1], model.outputDim.d[2], CV_8UC3);

    for(size_t x=0;x<coloredImg.rows;++x)
    {
        for(size_t y=0;y<coloredImg.cols;++y)
        {   
            int idx = x * coloredImg.cols + y;
            uint8_t label= (uint8_t)output[idx];

            if(label<20)
            {
                coloredImg.at<cv::Vec3b>(x,y)=colorMap[label];
            }
            else
            {
                coloredImg.at<cv::Vec3b>(x,y)=cv::Vec3b(0,0,0);
            }
        }
    }
    cv::imshow("mat", coloredImg);
    cv::waitKey(0);
    
    return 0;
}