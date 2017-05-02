/*
jcarron July 2016, Brighton.

Kernels for GPU calculation of lensing of CMB maps.
Use a GPU-based bicubic spline interpoaltion, using 2D texture fetching,
adapted freely from From CUDA toolkit set of examples, jcarron June 2016, Brighton.

All CUDA textures should be set here in normalised coordinates,POINT eval mode,
and set to wrapping mode to account for periodic boundary conditions.
*/
#include <pycuda-complex.hpp>
#include <math.h>
texture<float, 2> unl_CMB;
texture<float, 2> tex_dx;
texture<float, 2> tex_dy;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}
// filter 4 values using cubic splines
__device__ float cubicFilter(float x, float c0, float c1, float c2, float c3)
{
    float r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

__device__ int Freq(unsigned int i,unsigned int N)
{ // Frequency of rfft map for index i in map of shape N
return i - 2 * (i >= (N / 2)) * (i % (N / 2));
}

// slow but precise bicubic lookup using 16 texture lookups
__global__ void bicubiclensKernel(float* output,int width)
{

    // Calculate unnormalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions

    float _x = x + 0.5f;
    float _y = y + 0.5f;

    float dx = x +  tex2D(tex_dx,_x,_y) ;
    float dy = y +  tex2D(tex_dy,_x,_y) ;

    float px = floor(dx);
    float py = floor(dy);
    float fx = dx - px;
    float fy = dy - py;

    output[y * width + x] = cubicFilter(fy,
                          cubicFilter(fx, tex2D(unl_CMB, px-1, py-1), tex2D(unl_CMB, px, py-1), tex2D(unl_CMB, px+1, py-1), tex2D(unl_CMB, px+2,py-1)),
                          cubicFilter(fx, tex2D(unl_CMB, px-1, py),   tex2D(unl_CMB, px, py),   tex2D(unl_CMB, px+1, py),   tex2D(unl_CMB, px+2, py)),
                          cubicFilter(fx, tex2D(unl_CMB, px-1, py+1), tex2D(unl_CMB, px, py+1), tex2D(unl_CMB, px+1, py+1), tex2D(unl_CMB, px+2, py+1)),
                          cubicFilter(fx, tex2D(unl_CMB, px-1, py+2), tex2D(unl_CMB, px, py+2), tex2D(unl_CMB, px+1, py+2), tex2D(unl_CMB, px+2, py+2))
                         );
}
__global__ void detmagn_normtex(float* output,int width)
{
//  O(h**4) rule, here in grid units
// Calculate unnormalized texture coordinates

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions

    float _x = (x + 0.5f)/width;
    float _y = (y + 0.5f)/width;
    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;
    float h =  1.0f / width;
    float h2 = 2.0f / width;


    float dxdx = tex2D(tex_dx,_x - h2,_y) * w_2 + tex2D(tex_dx,_x - h,_y) * w_1 - tex2D(tex_dx,_x + h,_y) * w_1  - tex2D(tex_dx,_x + h2,_y) * w_2;
    float dxdy = tex2D(tex_dx,_x,_y - h2) * w_2 + tex2D(tex_dx,_x,_y - h) * w_1 - tex2D(tex_dx,_x,_y + h) * w_1  - tex2D(tex_dx,_x ,_y + h2)* w_2;
    float dydx = tex2D(tex_dy,_x - h2,_y) * w_2 + tex2D(tex_dy,_x - h,_y) * w_1 - tex2D(tex_dy,_x + h,_y) * w_1  - tex2D(tex_dy,_x + h2,_y) * w_2;
    float dydy = tex2D(tex_dy,_x,_y - h2) * w_2 + tex2D(tex_dy,_x,_y - h) * w_1 - tex2D(tex_dy,_x,_y + h) * w_1  - tex2D(tex_dy,_x ,_y + h2)* w_2;
    output[y * width + x] *= (1.0f + dxdx) * (1.0f + dydy) - (dydx * dxdy);

}
__global__ void bicubiclensKernel_normtex(float* output,int width)
{


    // Calculate unnormalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions

    float _x = x + 0.5f;
    float _y = y + 0.5f;

    float dx = x +  tex2D(tex_dx,_x / width,_y /width) ;
    float dy = y +  tex2D(tex_dy,_x / width,_y /width ) ;

    float px = floor(dx);
    float py = floor(dy);
    float fx = dx - px;
    float fy = dy - py;
    px /= width;
    py /= width;
    float f1_N = 1.0f / width;
    float f2_N = 2.0f / width;
    output[y * width + x] = cubicFilter(fy,
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py-f1_N), tex2D(unl_CMB, px, py-f1_N), tex2D(unl_CMB, px+f1_N, py-f1_N), tex2D(unl_CMB, px+f2_N,py-f1_N)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py),   tex2D(unl_CMB, px, py),   tex2D(unl_CMB, px+f1_N, py),   tex2D(unl_CMB, px+f2_N, py)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py+f1_N), tex2D(unl_CMB, px, py+f1_N), tex2D(unl_CMB, px+f1_N, py+f1_N), tex2D(unl_CMB, px+f2_N, py+f1_N)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py+f2_N), tex2D(unl_CMB, px, py+f2_N), tex2D(unl_CMB, px+f1_N, py+f2_N), tex2D(unl_CMB, px+f2_N, py+f2_N))
                         );
}
// Bilinear lensing interpolation
__global__ void bilinearlensKernel(float* output, int width)
{
       // Calculate unnormalized texture coordinates
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
       // Evaluation ref at displaced positions

       float _x = x + 0.5f;
       float _y = y + 0.5f;
       output[y * width + x] = tex2D(unl_CMB, _x + tex2D(tex_dx,_x,_y), _y + tex2D(tex_dy,_x,_y));
       //output[y * width + x] = tex2D(unl_CMB,_x,_y);

}
__global__ void bicubiclensKernel_normtex_singletex(float* output,float* _dx,float* _dy,int width)
{

    // Calculate unnormalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * width + x;

    float dx = x +  _dx[i] ;
    float dy = y +  _dy[i] ;

    float px = floor(dx);
    float py = floor(dy);
    float fx = dx - px;
    float fy = dy - py;
    float f1_N = 1.0f / width;
    float f2_N = 2.0f / width;

    px /= width;
    py /= width;
    output[i] = cubicFilter(fy,
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py-f1_N), tex2D(unl_CMB, px, py-f1_N), tex2D(unl_CMB, px+f1_N, py-f1_N), tex2D(unl_CMB, px+f2_N,py-f1_N)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py),   tex2D(unl_CMB, px, py),   tex2D(unl_CMB, px+f1_N, py),   tex2D(unl_CMB, px+f2_N, py)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py+f1_N), tex2D(unl_CMB, px, py+f1_N), tex2D(unl_CMB, px+f1_N, py+f1_N), tex2D(unl_CMB, px+f2_N, py+f1_N)),
                          cubicFilter(fx, tex2D(unl_CMB, px-f1_N, py+f2_N), tex2D(unl_CMB, px, py+f2_N), tex2D(unl_CMB, px+f1_N, py+f2_N), tex2D(unl_CMB, px+f2_N, py+f2_N))
                         );

}__device__ float tex2De(float* _unl_CMB,int i,int j,const unsigned int width)
{
    if ( (i >= width) || (i < 0) ) {i = i % width;} // periodicity
    if ( (j >= width) || (j < 0) ) {j = j % width;} // periodicity
    return _unl_CMB[j * width + i];
}
__global__ void bicubiclensKernel_notex(float* output,float* _unl_CMB,float* dx,float* dy,const unsigned int width)
{

    // Calculate unnormalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = y * width + x;

    float fx = x +  dx[z];
    float fy = y +  dy[z];

    int px = floor(fx);
    int py = floor(fy);
    fx -= px;
    fy -= py;

    output[z] = cubicFilter(fy,
                          cubicFilter(fx, tex2De(_unl_CMB, px -1, py -1,width), tex2De(_unl_CMB, px, py -1,width), tex2De(_unl_CMB, px+1, py -1,width), tex2De(_unl_CMB, px+2,py -1,width)),
                          cubicFilter(fx, tex2De(_unl_CMB, px -1, py,width),   tex2De(_unl_CMB, px, py,width),   tex2De(_unl_CMB, px+1, py,width),   tex2De(_unl_CMB, px+2, py,width)),
                          cubicFilter(fx, tex2De(_unl_CMB, px -1, py+1,width), tex2De(_unl_CMB, px, py+1,width), tex2De(_unl_CMB, px+1, py+1,width), tex2De(_unl_CMB, px+2, py+1,width)),
                          cubicFilter(fx, tex2De(_unl_CMB, px -1, py+2,width), tex2De(_unl_CMB, px, py+2,width), tex2De(_unl_CMB, px+1, py+2,width), tex2De(_unl_CMB, px+2, py+2,width))
                         );
}

__global__ void detmagn_notex(float* output,float* dx, float* dy,const unsigned int width)
{
//  O(h**4) rule, here in grid units
// Calculate unnormalized texture coordinates

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions


    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;


    float dxdx = tex2De(dx,x - 2,y,width) * w_2 + tex2De(dx,x - 1,y,width) * w_1 - tex2De(dx,x + 1,y,width) * w_1  - tex2De(dx,x + 2,y,width) * w_2;
    float dxdy = tex2De(dx,x,y - 2,width) * w_2 + tex2De(dx,x,y - 1,width) * w_1 - tex2De(dx,x,y + 1,width) * w_1  - tex2De(dx,x,y + 2,width) * w_2;
    float dydx = tex2De(dy,x - 2,y,width) * w_2 + tex2De(dy,x - 1,y,width) * w_1 - tex2De(dy,x + 1,y,width) * w_1  - tex2De(dy,x + 2,y,width) * w_2;
    float dydy = tex2De(dy,x,y - 2,width) * w_2 + tex2De(dy,x,y - 1,width) * w_1 - tex2De(dy,x,y + 1,width) * w_1  - tex2De(dy,x,y + 2,width) * w_2;
    output[y * width + x] *= (1.0f + dxdx) * (1.0f + dydy) - (dydx * dxdy);

}

// Bilinear lensing interpolation
__global__ void bilinearlensKernel_normtex_singletex(float* output, float* _dx, float* _dy,int width)
{
       // Calculate unnormalized texture coordinates
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
       unsigned int i = y * width + x;
       // Evaluation ref at displaced positions

       float _x = (x + _dx[i] + 0.5f)/width;
       float _y = (y + _dy[i] + 0.5f)/width;
       output[i] = tex2D(unl_CMB, _x,_y);

}
// Bilinear lensing interpolation
__global__ void bilinearlensKernel_normtex(float* output, int width)
{
       // Calculate unnormalized texture coordinates
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
       // Evaluation ref at displaced positions

       float _x = (x + 0.5f)/width;
       float _y = (y + 0.5f)/width;
       output[y * width + x] = tex2D(unl_CMB, _x + tex2D(tex_dx,_x,_y) / width, _y + tex2D(tex_dy,_x,_y) /width);
       //output[y * width + x] = tex2D(unl_CMB,_x,_y);

}
__global__ void cf_outer(pycuda::complex<float> *output, int width,int height)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
//    bicubic filter function


       if ((x < width) and (y < height))
        {
        float wx = 6.0f / (2.0f * cos(2.0f * M_PI * Freq(x,height) / height)  + 4.0f ) / height;
        float wy = 6.0f / (2.0f * cos(2.0f * M_PI * Freq(y,height) / height)  + 4.0f ) / height;
        output[y * width + x] *= wx * wy;
        }
}
__global__ void cf_outer_w(pycuda::complex<float> *output, float* w,int width,int height)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
//    bicubic filter function


       if ((x < width) and (y < height))
        {
        output[y * width + x] *= w[x] * w[y];
        }
}
__global__ void cdd_outer(pycuda::complex<double> *output, int width,int height)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
//    bicubic filter function

       if ((x < width) and (y < height))
        {
        double wx = 6.0 / (2.0 * cos(2.0 * M_PI * Freq(x,height) / height)  + 4.0 ) / height;
        double wy = 6.0 / (2.0 * cos(2.0 * M_PI * Freq(y,height) / height)  + 4.0 ) / height;

        output[y * width + x] *= wx * wy;
        }
}
__global__ void cdd_outer_w(pycuda::complex<double> *output,float* w, int width,int height)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//    wx = (6. / (2. * np.cos(2. * np.pi * Freq(np.arange(shape[0]), shape[0]) / shape[0]) + 4.) / shape[0])
//    bicubic filter function

       if ((x < width) and (y < height))
        {
        output[y * width + x] *= w[x] * w[y];
        }
}
__global__ void cf_mult(pycuda::complex<float> *output, float *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
__global__ void cf_add(pycuda::complex<float> *output, float *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] += b[x];
        }
}

__global__ void cdd_mult(pycuda::complex<double> *output, double *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
__global__ void cdd_add(pycuda::complex<double> *output, double *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] += b[x];
        }
}
__global__ void cc_mult(pycuda::complex<float> *output, pycuda::complex<float> *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
__global__ void ff_mult(float *output, float *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
__global__ void dcdc_mult(pycuda::complex<double> *output, pycuda::complex<double> *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
__global__ void dd_mult(double *output, double *b,int size)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

       if (x < size)
        {
        output[x] *= b[x];
        }
}
