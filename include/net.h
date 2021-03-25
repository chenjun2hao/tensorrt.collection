/**
 * @file net.h
 * @brief 
 * @author jun.chen (jun.chen@cloudminds.com)
 * @version 1.0
 * @date 2021-02-05
 * @copyright Copyright (c) 2021  cloudminds
 * @par 修改日志:
 *      1. 支持tensorrt7.2推理
 *      2. 不需要模型转换，模型转换采用onnx-tensorrt项目   
 */
#ifndef NET_H
#define NET_H

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#define VERSION "1.0.1"

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(error_code) << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

typedef unsigned char uchar;


namespace megEngine{
    using namespace std;

    enum class RUN_MODE
    {
        FLOAT32 = 0,
        FLOAT16 = 1,
        INT8    = 2,
    };

    struct InferDeleter {
      template<typename T>
      void operator()(T* obj) const {
        if( obj ) {
            obj->destroy();
        }
      }
  };

  template<typename T>
  inline std::shared_ptr<T> infer_object(T* obj) {
      if( !obj ) {
  	    throw std::runtime_error("Failed to create object");
      }
      return std::shared_ptr<T>(obj, InferDeleter());
  }


    class net
    {
        template <typename T>
        using nvUniquePtr = unique_ptr<T, InferDeleter>;

    private:
        shared_ptr<nvinfer1::ICudaEngine> mEngine;
        shared_ptr<nvinfer1::IExecutionContext> mContext;
        nvinfer1::IPluginFactory *mPlugin;
        cudaStream_t mCudaStream;
        vector<void *> mCudaBuffers;
        void *mCudaImg;
        void *demo_rgb;
        void *col_data;
        nvinfer1::DataType dtype;

    public:
        // parser tensorrt model
        net(const string &engineFile);
        // save tensorrt engine mode
        bool saveEngine(const string &fileName);
        bool initEngine();
        // image preprocess
        bool preprocess();
        // infer one image
        bool infer(const cv::Mat &img, void *output, cv::Mat &show);
        
        /**
         * @brief 
         * @param  input_data       My Param doc
         * @param  output           My Param doc
         * @param  showRgb          My Param doc
         * @return true 
         * @return false 
         */
        bool infer(float* input_data, void *output, cv::Mat &showRgb);
        
        ~net();

        static const int maxInputWidth;
        static const int maxInputHeight;
        static const int batchSize;
        static const bool visual;
        vector<size_t> mBindBufferSizes;
        nvinfer1::Dims outputDim;
        nvinfer1::Dims inputDim;
    };
    
    
    
}

#endif
