#include <numeric>
#include <cuda_runtime_api.h>
#include <fstream>
#include <assert.h>
#include "net.h"
#include "resize.h"
#include "common/logger.h"
#include <NvInferPlugin.h>          // 插件

const int megEngine::net::maxInputHeight = 4096;
const int megEngine::net::maxInputWidth  = 4096;
const int megEngine::net::batchSize      = 1;
const bool megEngine::net::visual        = true;


namespace megEngine{
    inline unsigned int getElementSize(nvinfer1::DataType t)
    {
        switch (t)
        {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        }
        throw runtime_error("Invalid datatype!");
        return 0;
    }

    inline int64_t volume(const nvinfer1::Dims &dim)
    {
        return accumulate(dim.d, dim.d + dim.nbDims, 1, multiplies<int64_t>());
    }

    inline void* safeMalloc(size_t size)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, size));
        if (deviceMem == nullptr)
        {
            std::cerr << "out of memery!";
            exit(1);
        }
        return deviceMem;
    }

    net::~net()
    {
        cudaStreamSynchronize(mCudaStream);
        cudaStreamDestroy(mCudaStream);
        if (mCudaImg) CUDA_CHECK(cudaFree(mCudaImg));
        if (mCudaBuffers[0]) CUDA_CHECK(cudaFree(mCudaBuffers[0]));
        if (mCudaBuffers[1]) CUDA_CHECK(cudaFree(mCudaBuffers[1]));
    }


    net::net(const string &engineFile)
    {
        CHECK(cudaSetDevice(0));

        std::ifstream file;
        file.open(engineFile, std::ios::binary | std::ios::in);
        if (!file.is_open())
        {
            std::cout << "open engine file:" << engineFile << "failed" << std::endl;
            return;
        }
        file.seekg(0, ios::end);
        int dataSize = file.tellg();
        file.seekg(0, ios::beg);
        unique_ptr<char[]> data(new char[dataSize]);
        file.read(data.get(), dataSize);
        file.close();

        auto mRunTime = infer_object( nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()) );
        assert(mRunTime != nullptr);

        initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");            // 注册插件，那python中应该有这样一句来初始化
        
        mEngine = infer_object( mRunTime->deserializeCudaEngine(data.get(), dataSize, nullptr) );
        assert(mEngine != nullptr);
        printf("load model successfully... \n");

        auto ret = initEngine();
        assert(ret); 
        printf("initEngine successfully! \n");     
    }


    bool net::initEngine()
    {
        std::cout << "---------------" << std::endl;
        mContext = infer_object( mEngine->createExecutionContext() );
        assert(mContext != nullptr);

        // input, output tensors
        int batchSize   = mEngine->getMaxBatchSize();
        int numBindings = mEngine->getNbBindings();
        mCudaBuffers.resize(numBindings);
        mBindBufferSizes.resize(numBindings);
        for (int i = 0; i < numBindings; i++)
        {
            dtype = mEngine->getBindingDataType(i);
            int memSize = batchSize * volume(mEngine->getBindingDimensions(i)) * getElementSize(dtype);
            mCudaBuffers[i] = safeMalloc(memSize);
            mBindBufferSizes[i] = memSize;
        }
        // get the input tensor dims
        inputDim  = mEngine->getBindingDimensions(0);
        outputDim = mEngine->getBindingDimensions(1);
        mCudaImg = safeMalloc( maxInputWidth * maxInputHeight * 3 * sizeof(uchar));

        // malloc for demo rgb image
        int dataSize = batchSize * volume(outputDim) * 3 * sizeof(uchar);
        demo_rgb = safeMalloc(dataSize);

        CUDA_CHECK(cudaStreamCreate(&mCudaStream));
        return 1;
    }


    bool net::infer(const cv::Mat &img, void *output, cv::Mat &showRgb)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCudaImg, img.data, img.step * img.rows, cudaMemcpyHostToDevice, mCudaStream));
        if (inputDim.d[3] == img.cols && inputDim.d[2] == img.rows)
            Norm(mCudaImg, (float*)mCudaBuffers[0], img.cols, img.rows, mCudaStream);
        else
            // resize and normlize image
            resizeAndNorm(mCudaImg, (float *)mCudaBuffers[0], img.cols, img.rows,
                        inputDim.d[3], inputDim.d[2], mCudaStream, 0, 0);
        

        // inference one image
        mContext->execute(batchSize, &mCudaBuffers[0]);

        int outDataSize = batchSize * volume(outputDim) * getElementSize(dtype);
        CUDA_CHECK(cudaMemcpyAsync(output, mCudaBuffers[1], outDataSize, cudaMemcpyDeviceToHost, mCudaStream));

        CHECK(cudaStreamSynchronize(mCudaStream));

        return 1;
    }

    bool net::infer(float* input_data, void *output, cv::Mat &showRgb)
    {
        CUDA_CHECK(cudaMemcpyAsync(mCudaBuffers[0], (void*)input_data, mBindBufferSizes[0], cudaMemcpyHostToDevice, mCudaStream));

        // inference one image
        mContext->execute(batchSize, &mCudaBuffers[0]);

        int outDataSize = batchSize * volume(outputDim) * getElementSize(dtype);
        CUDA_CHECK(cudaMemcpyAsync(output, mCudaBuffers[1], outDataSize, cudaMemcpyDeviceToHost, mCudaStream));

        CHECK(cudaStreamSynchronize(mCudaStream));

        return 1;
    }

}
