// yaoshun.li  2021-1-20

#ifndef _INFERENCE_H
#define _INFERENCE_H

#include <stdio.h>

struct Shape {
	Shape() { N = 1; C = 1; H = 1; W = 1; }
	Shape(unsigned int n, unsigned int c, unsigned int h, unsigned int w) {
		N = n; C = c; H = h; W = w;
	}
	unsigned int count() {
		return N * C * H * W;
	}
	unsigned int N;
	unsigned int C;
	unsigned int H;
	unsigned int W;
};

struct Datum {
	Datum() {
		data = nullptr;
		outter_data = nullptr;
	}
	float* Getdata() {
		if (data != nullptr) {
			return data;
		}
		else {
			unsigned int size = shape.count();
			data = new float[size]();
			return data;
		}
	}
	void Reshape(Shape s) {
		if (data != nullptr && s.count() > shape.count()) {
			delete[] data;
			data = nullptr;
		}
		shape = s;
	}
	~Datum() {
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
	}
	Shape shape;
	Shape out_shape;
	float* data = nullptr;//NCHW
	float* outter_data = nullptr;
};

enum class RUN_MODE
{
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT8    = 2,
};

class Inference {
public:
        explicit Inference(const char * engine_path);
		explicit Inference(const char * onnxFile, const char * calibFile, int maxBatchSize, RUN_MODE mode);
        int GetOutBufferSize();
        virtual ~Inference();
		void saveEngine(const char * fileName);
        void Infer(Datum &Dt, int flag);
private:
        void* InferNet_ = nullptr;

};
    


#endif //_INFERENCE_H
