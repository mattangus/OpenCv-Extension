#pragma once

#include "helpers.cuh"

#define ADD_FUNC(type)          \
    template __global__ void add<type>(type scalar, type* arr, int N)

#define DIVIDE_FUNC(type)       \
    template __global__ void divide<type>(type scalar, type* arr, int N)

#define MULTIPLY_FUNC(type)     \
    template __global__ void multiply<type>(type scalar, type* arr, int N)

#define FILL_FUNC(type)     \
    template __global__ void fill<type>(type value, type* arr, int N)

#define MAXTI_FUNC(type)    \
    template __global__ void maxTrackInds<type>(type* maxVals, type* toTest, int* inds, int testInd, int N)

#define ITOC_FUNC(type)     \
    template __global__ void indsToColour<type>(int* inds, type* maxVals, type* colours, int* r, int* g, int* b, int N)

#define CONV_FUNC(t1, t2)     \
    template __global__ void convertTo<t1, t2>(t1* src, t2* dest, int N)

#define DECLARE_FUNC(func)      \
    func(int);                  \
    func(float);                \
    func(double);               \
    func(char);


namespace ops
{
    template<typename dtype> __global__
    void divide(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] /= scalar;
        }
    }

    template<typename dtype> __global__
    void multiply(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] *= scalar;
        }
    }

    template<typename dtype> __global__
    void add(dtype scalar, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] += scalar;
        }
    }

    template<typename dtype> __global__
    void fill(dtype value, dtype* arr, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            arr[i] = value;
        }
    }

    template<typename dtype> __global__
    void maxTrackInds(dtype* maxVals, dtype* toTest, int* inds, int testInd, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(maxVals[i] < toTest[i])
            {
                maxVals[i] = toTest[i];
                inds[i] = testInd;
            }
        }
    }

    template<typename dtype> __global__
    void indsToColour(int* inds, dtype* maxVals, dtype* colours, int* r, int* g, int* b, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(maxVals[i/3] == 0)
                colours[i] = 0;
            else
            {
                if(i % 3 == 0)
                    colours[i] = (dtype)b[inds[i/3]];
                else if(i % 3 == 1)
                    colours[i] = (dtype)g[inds[i/3]];
                else if(i % 3 == 2)
                    colours[i] = (dtype)r[inds[i/3]];
            }
        }
    }

    template<typename dtype, typename target> __global__
    void convertTo(dtype* src, target* dest, int N)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            dest[i] = (target)src[i];
        }
    }

    template<typename dtype> __global__
    void mapColours(dtype* from, dtype* to, dtype* map, int N, int M)
    {
        CUDA_1D_KERNEL_LOOP(i, N)
        {
            if(i % 3 == 0)
            {
                int b = from[i];
                int g = from[i+1];
                int r = from[i+2];
                int ind = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
                if(ind < M)
                {
                    to[i] = map[ind];
                    to[i+1] = map[ind];
                    to[i+2] = map[ind];
                }
            }
        }
    }

    DECLARE_FUNC(MULTIPLY_FUNC);
    DECLARE_FUNC(DIVIDE_FUNC);
    DECLARE_FUNC(ADD_FUNC);
    DECLARE_FUNC(FILL_FUNC);
    DECLARE_FUNC(MAXTI_FUNC);
    DECLARE_FUNC(ITOC_FUNC);
    CONV_FUNC(int,float);
    CONV_FUNC(int,double);
    CONV_FUNC(int,char);
    CONV_FUNC(float,int);
    CONV_FUNC(float,double);
    CONV_FUNC(float,char);
    CONV_FUNC(double,int);
    CONV_FUNC(double,float);
    CONV_FUNC(double,char);
    CONV_FUNC(char,int);
    CONV_FUNC(char,float);
    CONV_FUNC(char,double);

}

#undef DECLARE_FUNC
#undef FILL_FUNC
#undef ADD_FUNC
#undef MULTIPLY_FUNC
#undef DIVIDE_FUNC
