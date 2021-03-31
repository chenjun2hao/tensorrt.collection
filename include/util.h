#include<vector>
#include<algorithm>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

/**
 * @brief  sort the output tensor
 * @param  tensor           classification output tensor
 * @param  dimensions       output tensor dims
 * @return std::vector<size_t> 
 */
std::vector<size_t> argsort(float *tensor, nvinfer1::Dims dimensions);

/**
 * @brief 
 * @param  dimensions       My Param doc
 * @return size_t 
 */
size_t numTensorElements(nvinfer1::Dims dimensions);

/**
 * @brief 
 * @param  image            My Param doc
 * @param  tensor           My Param doc
 * @param  dimensions       My Param doc
 */
void cvImageToTensor(const cv::Mat & image, float *tensor, nvinfer1::Dims dimensions);

/**
 * @brief 
 * @param  tensor           My Param doc
 * @param  dimensions       My Param doc
 * @param  mean             My Param doc
 * @param  std              My Param doc
 */
void preprocessVgg(float *tensor, nvinfer1::Dims dimensions, const float* mean, const float* std);