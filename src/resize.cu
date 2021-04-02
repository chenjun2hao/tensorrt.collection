#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "resize.h"

// in rgb order
__constant__ float mean[3] = {0.485, 0.456, 0.406};
__constant__ float Std[3]  = {0.229, 0.224, 0.225};

__forceinline__ __device__ float3 get(uchar3* src, int x,int y,int w,int h){
    if(x < 0 || x>=w || y<0 || y>=h) return make_float3(0.5,0.5,0.5);
    uchar3 temp = src[y*w + x];
    return make_float3(float(temp.x)/255.,float(temp.y)/255.,float(temp.z)/255.);
    // return make_float3(float(temp.x)/127.5,float(temp.y)/127.5,float(temp.z)/127.5);
}

__global__ void resizeNormKernel(uchar3* src,float *dst,int dstW, int dstH,int srcW,int srcH,
                                                float scaleX, float scaleY,float shiftX, float shiftY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = idx % dstW;
    const int y = idx / dstW;
    if (x >= dstW || y >= dstH)
        return;
    float w = (x - shiftX + 0.5) * scaleX - 0.5;        // 缩放的反向映射矩阵
    float h = (y - shiftY + 0.5) * scaleY - 0.5;        // opencv 
    int h_low = (int)h;
    int w_low = (int)w;
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float3 v1 = get(src,w_low,h_low,srcW,srcH);
    float3 v2 = get(src,w_high,h_low,srcW,srcH);
    float3 v3 = get(src,w_low,h_high,srcW,srcH);
    float3 v4 = get(src,w_high,h_high,srcW,srcH);
    int stride = dstW*dstH;
    // bgr -> rgb + normlization
    dst[y*dstW + x]            = ((w1 *v1.z + w2 * v2.z + w3 *v3.z + w4 * v4.z ) - mean[0]) / Std[0] ;
    dst[stride + y*dstW + x]   = ((w1 *v1.y + w2 * v2.y + w3 *v3.y + w4 * v4.y ) - mean[1]) / Std[1] ;
    dst[stride*2 + y*dstW + x] = ((w1 *v1.x + w2 * v2.x + w3 *v3.x + w4 * v4.x ) - mean[2]) / Std[2] ;
}

int resizeAndNorm(void * p,float *d,int w,int h,int in_w,int in_h, cudaStream_t stream, bool keepration ,bool keepcenter){
    float scaleX = (w*1.0f / in_w);
    float scaleY = (h*1.0f / in_h);
    float shiftX = 0.f ,shiftY = 0.f;
    if(keepration)scaleX = scaleY = scaleX > scaleY ? scaleX : scaleY;
    if(keepration && keepcenter){shiftX = (in_w - w/scaleX)/2.f;shiftY = (in_h - h/scaleY)/2.f;}
    const int n = in_w*in_h;
    int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;
    resizeNormKernel<<<gridSize, blockSize, 0, stream>>>((uchar3*)(p),d,in_w,in_h,w,h,scaleX,scaleY,shiftX,shiftY);
    return 0;
}

__global__ void NormKernel(uchar3* src, float* dst, int w, int h, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    int x = idx % w;
    int y = idx / w;
    float3 v = get(src, x, y, w, h);
    int stride = w * h;
    dst[stride * 0 + y * w + x] = (v.z - mean[0]) / Std[0];
    dst[stride * 1 + y * w + x] = (v.y - mean[1]) / Std[1];
    dst[stride * 2 + y * w + x] = (v.x - mean[2]) / Std[2];
}

int Norm(void* src, float* dst, int w, int h, cudaStream_t stream){
    int n = w * h;
    int blockSize = 512;
    int gridSize  = (n + blockSize - 1) / blockSize;
    NormKernel<<<gridSize, blockSize, 0, stream>>>((uchar3*)(src), dst, w, h, n);
    return 0;
}

// in rgb order
const int num_cls = 17;
__constant__ unsigned char map_[num_cls][3] = { {0, 0, 0},
                                                {0, 0, 255},
                                                {0, 255, 0},
                                                {0, 255, 255},
                                                {255, 0, 0 },
                                                {255, 0, 255 }, 
                                                {255, 255, 0 },
                                                {255, 255, 255 },
                                                {0, 0, 128 },
                                                {0, 128, 0 },
                                                {0, 128, 128 },
                                                {128, 0, 0 },
                                                {128, 0, 128 },
                                                {128, 128, 0 },
                                                {128, 128, 128 },
                                                {192, 192, 192 },
                                                {0,   255, 0},
                                             };

// transfer the prob map to the rgb image on the GPU
__global__ void prob2rgb_kernel(const int * const arg_data, uchar3 *ori, unsigned char * const rgb, const int N)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * blockDim.x * gridDim.x;
    if(tid < N)     // 判断是否超限
    {
        int maxIdx = 0;
        maxIdx = arg_data[tid];
        uchar3 temp = ori[y * 640 + x];

        unsigned char t0 = (map_[maxIdx][2] * 0.5 + temp.x * 0.5) > 255 ? 255 : (map_[maxIdx][2] * 0.5 + temp.x * 0.5);
        unsigned char t1 = (map_[maxIdx][1] * 0.5 + temp.y * 0.5) > 255 ? 255 : (map_[maxIdx][1] * 0.5 + temp.y * 0.5);
        unsigned char t2 = (map_[maxIdx][0] * 0.5 + temp.z * 0.5) > 255 ? 255 : (map_[maxIdx][0] * 0.5 + temp.z * 0.5);
        rgb[tid * 3 + 0] = t0;
        rgb[tid * 3 + 1] = t1;
        rgb[tid * 3 + 2] = t2;
    }
}


/*
* brief  drar rgb image for sematic result
* @param  arg_data  the predict target label
* @param  rgb       the result rgb image   
*/
int prob2rgb(const int * const arg_data, void *ori, unsigned char * const rgb, const int w, const int h, cudaStream_t stream)
{
    const int threads_per_block = 16;      
    dim3 blocks = dim3((w + threads_per_block - 1) / threads_per_block, (h  + threads_per_block - 1) / threads_per_block);
    dim3 threads = dim3(threads_per_block, threads_per_block);
    int size = w * h;
    prob2rgb_kernel<<<blocks, threads, 0, stream>>>(arg_data, (uchar3 *)ori, rgb, size);

    return 0;
}
