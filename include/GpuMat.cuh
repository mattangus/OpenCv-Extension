#pragma once

#include <cuda.h>
#include <opencv2/opencv.hpp>

#include "helpers.cuh"
#include "ops.cuh"
#include "GpuVector.cuh"

template<typename dtype>
class GpuMat
{
private:
    const static int kThreadsPerBlock = 1024;
public:
    int height, width, depth, numElem, size;
    dtype* gpu_data;

    /**
     * Initialize a gpu mat with value "value"
     */
    GpuMat(int height, int width, int depth, dtype value) : GpuMat(height, width, depth, false)
    {
        fill(value);
    }

    /**
     * Initialize a gpu mat with the data from an opencv matrix
     */
    GpuMat(cv::Mat& mat) : GpuMat(mat.rows, mat.cols, mat.channels(), false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, mat.ptr(), size, cudaMemcpyHostToDevice) );
    }

    GpuMat(GpuMat<dtype>& other) : GpuMat(other.height, other.width, other.depth, false)
    {
        gpuErrchk( cudaMemcpy(gpu_data, other.gpu_data, size, cudaMemcpyDeviceToDevice) );
    }

    /**
     * Initialize a gpu mat with zeros
     */
    GpuMat(int height, int width, int depth, bool zero = true)
    {
        this->height = height;
        this->width = width;
        this->depth = depth;
        numElem = height*width*depth;
        size = numElem*sizeof(dtype);

        gpuErrchk( cudaMalloc((void**) &gpu_data, size) );
        if(zero)
            gpuErrchk( cudaMemset( gpu_data, 0, size) );	
    }

    ~GpuMat()
    {
        gpuErrchk( cudaFree(gpu_data) ); 
    }

    cv::Mat getMat()
    {
        dtype* output_im = new dtype[numElem];
        gpuErrchk( cudaMemcpy(output_im, gpu_data, size, cudaMemcpyDeviceToHost) );
        std::vector<int> sizes = {height, width, depth};
        //clone so opencv owns data
        cv::Mat ret = cv::Mat(height, width, CV_MAKETYPE(cv::DataType<dtype>::type, depth), output_im).clone();
        delete output_im;
        return ret;
    }

    template <typename target>
    void convertTo(GpuMat<target>& targetMat)
    {
        if(targetMat.height != height || targetMat.width != width || targetMat.depth != depth)
            throw std::runtime_error("targetMat must have same height, width and depth as source");
        LAUNCH(SINGLE_ARG(ops::convertTo<dtype, target>))(gpu_data, targetMat.gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void divide(dtype scalar)
    {
        LAUNCH(ops::divide<dtype>)(scalar, gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

    }
        
    void multiply(dtype scalar)
    {
        LAUNCH(ops::multiply<dtype>)(scalar, gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
        
    void add(dtype scalar)
    {
        LAUNCH(ops::add<dtype>)(scalar, gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
        
    void fill(dtype value)
    {
        LAUNCH(ops::fill<dtype>)(value, gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void load(cv::Mat& other)
    {
        if(other.rows != height || other.cols != width || other.channels() != depth)
            throw std::runtime_error("Loading Mats that don't have the same height width and depth is not supported");
        gpuErrchk( cudaMemcpy(gpu_data, other.ptr(), size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void maxTrackInds(GpuMat<dtype>& other, GpuMat<int>* inds, int otherInd)
    {
        if(numElem != other.numElem || numElem != inds->numElem || other.numElem != inds->numElem)
            throw std::runtime_error("Number of elements must match");
        LAUNCH(ops::maxTrackInds<dtype>)(gpu_data, other.gpu_data, inds->gpu_data, otherInd, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    void indsToColour(GpuMat<int>& inds, GpuMat<dtype>& maxVals, GpuVector<int>& r, GpuVector<int>& g, GpuVector<int>& b)
    {
        if(inds.width != width || inds.height != height || depth != 3 ||
            inds.depth != 1 || b.numElem != r.numElem ||
            r.numElem != g.numElem || g.numElem != b.numElem || inds.size != maxVals.size)
        {
            std::cout << inds.width << ", " << width << ", " <<  inds.height << ", " <<  height << ", " <<  depth
                << ", " << inds.depth << ", " <<  inds.numElem << ", " <<  r.numElem 
                << ", " << inds.numElem << ", " <<  g.numElem << ", " <<  inds.numElem << ", " <<  b.numElem << std::endl;
            throw std::runtime_error("Invalid input");
        }
        LAUNCH(ops::indsToColour<dtype>)(inds.gpu_data, maxVals.gpu_data, this->gpu_data, r.gpu_data, g.gpu_data, b.gpu_data, numElem);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
};