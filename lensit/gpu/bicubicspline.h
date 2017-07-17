#include <math.h>
/*
jcarron July 2016, Brighton.

Bicubic spline interpolation for lensing of CMB maps.
Adapted from GPU implementation of bicubic spline interpolation, using originally 2D texture fetching.
Adapted freely from From CUDA toolkit set of examples, jcarron June 2016, Brighton.

Starting from real space maps loaded in memory, performance in serial called from python looks roughly like :
* 33   Megapix /s (excl. prefiltering) if the map already prefiltered.
* 8.7  Megapix /s (incl prefiltering with numpy.fft) to lens a map, where the cost comes from the ffts.
(e.g. if need to lens the same map with diverse displacement field).

This is how to call it from python :

import weave
code = r"\
        for( unsigned int j= 0; j < width; j++ )\
        {\
        for( unsigned int i = 0; i < width; i++)\
            {\
            output[j * width + i] = bicubiclensKernel(prefilteredunl_CMB,i + dx[j * width + i],j + dy[j * width + i],width);\
            }\
         }"

weave.inline(code,['output','prefilteredunl_CMB','dx','dy','width'], headers=[ r' "bicubicspline.h" '])
*/

// w0, w1, w2, and w3 are the four cubic B-spline basis functions (times a factor 6)
template<typename T>
T w0(const T a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return a*(a*(-a + 3.0) - 3.0) + 1.0;
}
template<typename T>
T w1(const T a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return a*a*(3.0*a - 6.0) + 4.0;
}
template<typename T>
T w2(const T a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return a*(a*(-3.0*a + 3.0) + 3.0) + 1.0;
}
template<typename T>
T w3(const T a)
{
    return a*a*a;
}
// filter 4 values using cubic splines
template<typename T>
T cubicFilter(const T x, const T c0, const T c1, const T c2, const T c3)
{
    return (c0 * w0(x) + c1 * w1(x) + c2 * w2(x) + c3 * w3(x)) * (1.0/6.0);
}
template<typename T>
T tex2D(const T* unl_CMB,int i,int j,const unsigned int width)
{
    if ( (i >= width) || (i < 0) ) {i = i % width;} // periodicity
    if ( (j >= width) || (j < 0) ) {j = j % width;} // periodicity
    return unl_CMB[j * width + i];
}
template<typename T>
T tex2D_rect(const T* unl_CMB,int ix,int jy,const unsigned int width,const unsigned int height)
{
    if ( (jy >= height) || (jy < 0) ) {jy = jy % height;} // periodicity
    if ( (ix >= width) || (ix < 0) )  {ix= ix % width;} // periodicity
    return unl_CMB[jy * width + ix];
}
template<typename T>
T bicubiclensKernel(const T* unl_CMB,T fx, T fy,const unsigned int width)
{

    int px = floor(fx);
    int py = floor(fy);
    fx -= px;
    fy -= py;

    return  cubicFilter(fy,
            cubicFilter(fx, tex2D(unl_CMB, px-1, py-1,width), tex2D(unl_CMB, px, py-1,width), tex2D(unl_CMB, px+1, py-1,width), tex2D(unl_CMB, px+2,py-1,width)),
            cubicFilter(fx, tex2D(unl_CMB, px-1, py,width),   tex2D(unl_CMB, px, py,width),   tex2D(unl_CMB, px+1, py,width),   tex2D(unl_CMB, px+2, py,width)),
            cubicFilter(fx, tex2D(unl_CMB, px-1, py+1,width), tex2D(unl_CMB, px, py+1,width), tex2D(unl_CMB, px+1, py+1,width), tex2D(unl_CMB, px+2, py+1,width)),
            cubicFilter(fx, tex2D(unl_CMB, px-1, py+2,width), tex2D(unl_CMB, px, py+2,width), tex2D(unl_CMB, px+1, py+2,width), tex2D(unl_CMB, px+2, py+2,width))
                        );
}
template<typename T>
T bicubiclensKernel_rect(const T* unl_CMB,T fx, T fy,const unsigned int w,const unsigned int h)
{
    // w : second dimension of array  ('x')
    // h : first dimension of array  ('y')
    int px = floor(fx);
    int py = floor(fy);
    fx -= px;
    fy -= py;

    return  cubicFilter(fy,
            cubicFilter(fx, tex2D_rect(unl_CMB, px-1, py-1,w,h), tex2D_rect(unl_CMB, px, py-1,w,h), tex2D_rect(unl_CMB, px+1, py-1,w,h), tex2D_rect(unl_CMB, px+2,py-1,w,h)),
            cubicFilter(fx, tex2D_rect(unl_CMB, px-1, py,w,h),   tex2D_rect(unl_CMB, px, py,w,h),   tex2D_rect(unl_CMB, px+1, py,w,h),   tex2D_rect(unl_CMB, px+2, py,w,h)),
            cubicFilter(fx, tex2D_rect(unl_CMB, px-1, py+1,w,h), tex2D_rect(unl_CMB, px, py+1,w,h), tex2D_rect(unl_CMB, px+1, py+1,w,h), tex2D_rect(unl_CMB, px+2, py+1,w,h)),
            cubicFilter(fx, tex2D_rect(unl_CMB, px-1, py+2,w,h), tex2D_rect(unl_CMB, px, py+2,w,h), tex2D_rect(unl_CMB, px+1, py+2,w,h), tex2D_rect(unl_CMB, px+2, py+2,w,h))
                        );
}
template<typename T>
void lens_prefilteredmap(T* output, const T* unl_CMB,const T* dx,const T* dy,const unsigned int width)
{ // dx,dy displacement in pixel units.
   for( unsigned int j= 0; j < width; j++ )
        {
        for( unsigned int i = 0; i < width; i++)
            {
            output[j * width + i] = bicubiclensKernel(unl_CMB,i + dx[j * width + i],j + dy[j * width + i],width);\
            }
         }
}