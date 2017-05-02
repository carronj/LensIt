/*
jcarron July 2016, Brighton.

Kernels for GPU calculation of CMB lensing inverse displacement fields.
This file for pure potential displacement field, using only three textures
for the magnofication.
Otherwise similar to kernel inversion. Still uses dx and y textures as both of these must be lensed
at some point. Uses a NR iteration scheme, where the input displacement and magnification
are interpolated using a GPU-optimized bicubic spline algorithm, just like lensing.

All CUDA textures should be set here in normalised coordinates,POINT eval mode,
and set to wrapping mode to account for periodic boundary conditions.
*/
#include <math.h>
#include <pycuda-complex.hpp>
texture<float, 2> tex_dx;  // dx displacement, in grid units
texture<float, 2> tex_dy;  // dx displacement, in grid units

texture<float, 2> Minv_xx; // negative inverse magnification, xx elements
texture<float, 2> Minv_yy; // negative inverse magnification, xy elements
texture<float, 2> Minv_xy; // negative inverse magnification, yy elements


__device__ int sgndFreq(unsigned int i,unsigned int N)
{ // Frequency of rfft map for index i in map of shape N
int sgn = i >= (N / 2) ? -1 : 1;
return sgn * (i - 2 * (i >= (N / 2)) * (i % (N / 2)));
}

__global__ void cf_outer_w(pycuda::complex<float> *output, float* w,int width,int height)
{
// For the prefiltering
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       if ((x < width) and (y < height))
        {
        output[y * width + x] *= w[x] * w[y];
        }
}
__global__ void mult_rfft_inplace(pycuda::complex<float> *output, float w,int width,int height)
{
// For the prefiltering
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       if ((x < width) and (y < height))
        {
        output[y * width + x] *= w;
        }
}
__global__ void cc_outer_ikx(pycuda::complex<float> *output,pycuda::complex<float> *a,int width,int height)
{
// For the prefiltering
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       if ((x < width) and (y < height))
        {
        pycuda::complex<float> ikx(0.f,sgndFreq(x,height) * M_PI * 2.0f / height);
        output[y * width + x] = a[y * width + x] * ikx;
        }
}
__global__ void cc_outer_iky_inplace(pycuda::complex<float> *output,int width,int height)
{
// For the prefiltering
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

       if ((x < width) and (y < height))
        {
        pycuda::complex<float> iky(0.f,sgndFreq(y,height) * M_PI * 2.0f / height);
        output[y * width + x] *= iky;
        }
}
__global__ void ff_mult_inplace(float *output, float b, int width)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
       if ((x < width) and (y < width))
         {
         output[y * width + x] *= b;
         }
}
__global__ void ff_mult(float *output, float *a, float b, int width)
{
       unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
       unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
       if ((x < width) and (y < width))
         {
         output[y * width + x] = a[y * width + x] * b;
         }
}
// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
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
__global__ void divide_detmagn(float* output,int width)
{
//  Division by determinant of magnification  matrix
//  O(h**4) rule, here in grid units.

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions
    if ((x < width) and (y < width))
    {
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
    output[y * width + x] /= ( (1.0f + dxdx) * (1.0f + dydy) - dydx * dxdy );
      }
}

__global__ void get_m1pMxx(float* output,int width)
{
//  O(h**4) rule, here in grid units.
// -(1 + dx/dx).

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions
    if ((x < width) and (y < width))
    {

    float _x = (x + 0.5f)/width;
    float _y = (y + 0.5f)/width;
    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;
    float h =  1.0f / width;
    float h2 = 2.0f / width;

    output[y * width + x] = -1.0f - tex2D(tex_dx,_x - h2,_y) * w_2
                                  - tex2D(tex_dx,_x - h ,_y) * w_1
                                  + tex2D(tex_dx,_x + h ,_y) * w_1
                                  + tex2D(tex_dx,_x + h2,_y) * w_2;
    }
}
__global__ void get_m1pMyy(float* output,int width)
{
//  O(h**4) rule, here in grid units
// 1 + dy/dy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions
    if ((x < width) and (y < width))
    {
    float _x = (x + 0.5f)/width;
    float _y = (y + 0.5f)/width;
    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;
    float h =  1.0f / width;
    float h2 = 2.0f / width;

    output[y * width + x] = -1.0f - tex2D(tex_dy,_x,_y - h2) * w_2
                                  - tex2D(tex_dy,_x,_y - h)  * w_1
                                  + tex2D(tex_dy,_x,_y + h)  * w_1
                                  + tex2D(tex_dy,_x,_y + h2) * w_2;
    }
}
__global__ void get_Mxy(float* output,int width)
{
//  O(h**4) rule, here in grid units
// - dx /dy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions
    if ((x < width) and (y < width))
    {

    float _x = (x + 0.5f)/width;
    float _y = (y + 0.5f)/width;
    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;
    float h =  1.0f / width;
    float h2 = 2.0f / width;

    output[y * width + x] =      +  tex2D(tex_dx,_x ,_y -h2) * w_2
                                 +  tex2D(tex_dx,_x ,_y -h)  * w_1
                                 -  tex2D(tex_dx,_x ,_y + h) * w_1
                                 -  tex2D(tex_dx,_x ,_y + h2)* w_2;
    }
}
__global__ void get_Myx(float* output,int width)
{
//  O(h**4) rule, here in grid units
// - dy /dx

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Evaluation ref at displaced positions
    if ((x < width) and (y < width))
    {

    float _x = (x + 0.5f) / width;
    float _y = (y + 0.5f) / width;
    float w_2 = 0.08333333333f; // 1/12
    float w_1 = -0.66666666666f;
    float h =  1.0f / width;
    float h2 = 2.0f / width;


    output[y * width + x] =      + tex2D(tex_dy,_x - h2,_y) * w_2
                                 + tex2D(tex_dy,_x - h ,_y) * w_1
                                 - tex2D(tex_dy,_x + h ,_y) * w_1
                                 - tex2D(tex_dy,_x + h2,_y) * w_2;
    }
}
__global__ void displ_0th(float* ex,float* ey,int width)
{
// First Iteration of the displacement inversion, e_1 = - M_f^{-1}(y) * d(y)
// Just fetching textures

    // Normalised texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < width) and (y < width))
    {
    float _x = (x + 0.5f) / width;
    float _y = (y + 0.5f) / width;

    unsigned int i = y * width + x;

    ex[i] = tex2D(Minv_xx,_x,_y) * tex2D(tex_dx,_x,_y) + tex2D(Minv_xy,_x,_y) * tex2D(tex_dy,_x,_y);
    ey[i] = tex2D(Minv_xy,_x,_y) * tex2D(tex_dx,_x,_y) + tex2D(Minv_yy,_x,_y) * tex2D(tex_dy,_x,_y) ;
    }
}
__global__ void iterate(float* ex,float* ey,int width)
{
// Iterates the displacement inversion, e_N+1 = e_N - M_f^{-1}(y + e_N) * (e_N + d(y + e_N))
// For this we need the lens the 4 magnification matrix entries plus the two displacement components.

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x < width) and (y < width))
    {

    unsigned int i = y * width + x;
    float dx = x +  ex[i];
    float dy = y +  ey[i];

    float px = floor(dx);
    float py = floor(dy);
    float fx = dx - px;
    float fy = dy - py;
    float f1_N = 1.0f / width;
    float f2_N = 2.0f / width;

    px /= width;
    py /= width;


    float ex_len_dx =    ex[i] +   cubicFilter(fy,
                     cubicFilter(fx, tex2D(tex_dx, px-f1_N, py-f1_N), tex2D(tex_dx, px, py-f1_N), tex2D(tex_dx, px+f1_N, py-f1_N), tex2D(tex_dx, px+f2_N,py-f1_N)),
                     cubicFilter(fx, tex2D(tex_dx, px-f1_N, py),      tex2D(tex_dx, px, py),      tex2D(tex_dx, px+f1_N, py),      tex2D(tex_dx, px+f2_N, py)),
                     cubicFilter(fx, tex2D(tex_dx, px-f1_N, py+f1_N), tex2D(tex_dx, px, py+f1_N), tex2D(tex_dx, px+f1_N, py+f1_N), tex2D(tex_dx, px+f2_N, py+f1_N)),
                     cubicFilter(fx, tex2D(tex_dx, px-f1_N, py+f2_N), tex2D(tex_dx, px, py+f2_N), tex2D(tex_dx, px+f1_N, py+f2_N), tex2D(tex_dx, px+f2_N, py+f2_N))
                                                );
    float ey_len_dy =    ey[i] +   cubicFilter(fy,
                     cubicFilter(fx, tex2D(tex_dy, px-f1_N, py-f1_N), tex2D(tex_dy, px, py-f1_N), tex2D(tex_dy, px+f1_N, py-f1_N), tex2D(tex_dy, px+f2_N,py-f1_N)),
                     cubicFilter(fx, tex2D(tex_dy, px-f1_N, py),      tex2D(tex_dy, px, py),      tex2D(tex_dy, px+f1_N, py),      tex2D(tex_dy, px+f2_N, py)),
                     cubicFilter(fx, tex2D(tex_dy, px-f1_N, py+f1_N), tex2D(tex_dy, px, py+f1_N), tex2D(tex_dy, px+f1_N, py+f1_N), tex2D(tex_dy, px+f2_N, py+f1_N)),
                     cubicFilter(fx, tex2D(tex_dy, px-f1_N, py+f2_N), tex2D(tex_dy, px, py+f2_N), tex2D(tex_dy, px+f1_N, py+f2_N), tex2D(tex_dy, px+f2_N, py+f2_N))
                                                );
    float len_Mxx =       cubicFilter(fy,
                     cubicFilter(fx, tex2D(Minv_xx, px-f1_N, py-f1_N), tex2D(Minv_xx, px, py-f1_N), tex2D(Minv_xx, px+f1_N, py-f1_N), tex2D(Minv_xx, px+f2_N,py-f1_N)),
                     cubicFilter(fx, tex2D(Minv_xx, px-f1_N, py),      tex2D(Minv_xx, px, py),      tex2D(Minv_xx, px+f1_N, py),      tex2D(Minv_xx, px+f2_N, py)),
                     cubicFilter(fx, tex2D(Minv_xx, px-f1_N, py+f1_N), tex2D(Minv_xx, px, py+f1_N), tex2D(Minv_xx, px+f1_N, py+f1_N), tex2D(Minv_xx, px+f2_N, py+f1_N)),
                     cubicFilter(fx, tex2D(Minv_xx, px-f1_N, py+f2_N), tex2D(Minv_xx, px, py+f2_N), tex2D(Minv_xx, px+f1_N, py+f2_N), tex2D(Minv_xx, px+f2_N, py+f2_N))
                                                );

    float len_Myy =       cubicFilter(fy,
                     cubicFilter(fx, tex2D(Minv_yy, px-f1_N, py-f1_N), tex2D(Minv_yy, px, py-f1_N), tex2D(Minv_yy, px+f1_N, py-f1_N), tex2D(Minv_yy, px+f2_N,py-f1_N)),
                     cubicFilter(fx, tex2D(Minv_yy, px-f1_N, py),      tex2D(Minv_yy, px, py),      tex2D(Minv_yy, px+f1_N, py),      tex2D(Minv_yy, px+f2_N, py)),
                     cubicFilter(fx, tex2D(Minv_yy, px-f1_N, py+f1_N), tex2D(Minv_yy, px, py+f1_N), tex2D(Minv_yy, px+f1_N, py+f1_N), tex2D(Minv_yy, px+f2_N, py+f1_N)),
                     cubicFilter(fx, tex2D(Minv_yy, px-f1_N, py+f2_N), tex2D(Minv_yy, px, py+f2_N), tex2D(Minv_yy, px+f1_N, py+f2_N), tex2D(Minv_yy, px+f2_N, py+f2_N))
                                                );
    float len_Mxy =       cubicFilter(fy,
                     cubicFilter(fx, tex2D(Minv_xy, px-f1_N, py-f1_N), tex2D(Minv_xy, px, py-f1_N), tex2D(Minv_xy, px+f1_N, py-f1_N), tex2D(Minv_xy, px+f2_N,py-f1_N)),
                     cubicFilter(fx, tex2D(Minv_xy, px-f1_N, py),      tex2D(Minv_xy, px, py),      tex2D(Minv_xy, px+f1_N, py),      tex2D(Minv_xy, px+f2_N, py)),
                     cubicFilter(fx, tex2D(Minv_xy, px-f1_N, py+f1_N), tex2D(Minv_xy, px, py+f1_N), tex2D(Minv_xy, px+f1_N, py+f1_N), tex2D(Minv_xy, px+f2_N, py+f1_N)),
                     cubicFilter(fx, tex2D(Minv_xy, px-f1_N, py+f2_N), tex2D(Minv_xy, px, py+f2_N), tex2D(Minv_xy, px+f1_N, py+f2_N), tex2D(Minv_xy, px+f2_N, py+f2_N))
                                                );

    ex[i] += len_Mxx * ex_len_dx + len_Mxy * ey_len_dy;
    ey[i] += len_Mxy * ex_len_dx + len_Myy * ey_len_dy;
   }
}
