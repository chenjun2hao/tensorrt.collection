#include "util.h"

size_t numTensorElements(nvinfer1::Dims dimensions)
{
  if (dimensions.nbDims == 0)
    return 0;
  size_t size = 1;
  for (int i = 0; i < dimensions.nbDims; i++)
    size *= dimensions.d[i];
  return size;
}

std::vector<size_t> argsort(float *tensor, nvinfer1::Dims dimensions)
{
  size_t numel = numTensorElements(dimensions);
  std::vector<size_t> indices(numel);
  for (int i = 0; i < numel; i++)
    indices[i] = i;
  std::sort(indices.begin(), indices.begin() + numel, [tensor](size_t idx1, size_t idx2) {
      return tensor[idx1] > tensor[idx2];
  });

  return indices;
}

void cvImageToTensor(const cv::Mat & image, float *tensor, nvinfer1::Dims dimensions)
{
  const size_t channels = dimensions.d[1];
  const size_t height = dimensions.d[2];
  const size_t width = dimensions.d[3];
  // TODO: validate dimensions match
  const size_t stridesCv[3] = { width * channels, channels, 1 };
  const size_t strides[3] = { height * width, width, 1 };

  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      for (int k = 0; k < channels; k++) 
      {
        const size_t offsetCv = i * stridesCv[0] + j * stridesCv[1] + k * stridesCv[2];
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = (float) image.data[offsetCv];
      }
    }
  }
}

void preprocessVgg(float *tensor, nvinfer1::Dims dimensions, const float* mean, const float* std)
{
  size_t channels = dimensions.d[1];
  size_t height = dimensions.d[2];
  size_t width = dimensions.d[3];
  const size_t strides[3] = { height * width, width, 1 };
  // const float mean[3] = { 0.485, 0.456, 0.406 }; // values from TensorFlow slim models code
  // const float std[3]  = {0.229, 0.224, 0.225};

  for (int i = 0; i < height; i++) 
  {
    for (int j = 0; j < width; j++) 
    {
      for (int k = 0; k < channels; k++) 
      {
        const size_t offset = k * strides[0] + i * strides[1] + j * strides[2];
        tensor[offset] = ((tensor[offset] / 255.0) - mean[k]) / std[k];
      }
    }
  }
}