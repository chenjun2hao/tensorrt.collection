#ifndef RESIZE_H
#define RESIZE_H
/*
* @brief     inter linear resize and normlization
* 
*/
int resizeAndNorm(void *src, float *dst, int w, int h, int in_w, int in_h, cudaStream_t stream, bool keepration=0, bool keepcenter=0);

/**
 * @brief 
 * @param  src              My Param doc
 * @param  dst              My Param doc
 * @param  w                My Param doc
 * @param  h                My Param doc
 * @param  stream           My Param doc
 * @return int 
 */
int Norm(void* src, float* dst, int w, int h, cudaStream_t stream);

/*
* @brief     inter linear resize and normlization
* arg_data   the class label
* rgb        demo show rgb image
* stream     cuda stream
*/
int prob2rgb(const int * const arg_data, void *ori, unsigned char * const rgb, const int w, const int h, cudaStream_t stream);


/*
* @brief     im2col operater
* arg_data   the class label
* data_col   the transform data with (h*w) * (kernel_w * kernel_h)
* kernel_size the kernel's size
* stream     cuda stream
*/

void dilate(int *arg_data, void *data_col, int kernel_size, int w, int h, cudaStream_t stream);
void dilate(float *arg_data, void *data_col, int kernel_size, int w, int h, cudaStream_t stream);
#endif
