#include "bicubicspline.h"
template<typename T>
void displ_0th(T* ex,T* ey,const T* dx, const T* dy,const T* Minv_xx, const T* Minv_yy,const T* Minv_xy, const T* Minv_yx,int width)
{
// First Iteration of the displacement inversion, e_1 = - M_f^{-1}(y) * d(y)
// Just fetching textures

    for( unsigned int z = 0; z < width * width; z++ )
           {
            ex[z] = Minv_xx[z] * dx[z]  + Minv_xy[z] * dy[z];
            ey[z] = Minv_yx[z] * dx[z]  + Minv_yy[z] * dy[z] ;
            }
}
template<typename T>
void iterate(T* ex,T* ey,const T* dx, const T* dy,const T* Minv_xx, const T* Minv_yy,const T* Minv_xy, const T* Minv_yx,int width)
{
// Iterates the displacement inversion, e_N+1 = e_N - M_f^{-1}(y + e_N) * (e_N + d(y + e_N))
// For this we need the lens the 4 magnification matrix entries plus the two displacement components.
// The magnification matrix and displacement must be bicubic prefiltered.

    T fx,fy;
    T ex_len_dx,ey_len_dy,len_Mxx,len_Mxy,len_Myx,len_Myy;
    unsigned int i;

    for( unsigned int y= 0; y < width; y++ )
       {
       for( unsigned int x = 0; x < width; x++)
        {
        i = y * width + x;
        fx = x +  ex[i];
        fy = y +  ey[i];
        ex_len_dx = ex[i] +  bicubiclensKernel(dx,fx,fy,width);
        ey_len_dy = ey[i] +  bicubiclensKernel(dy,fx,fy,width);
        len_Mxx =  bicubiclensKernel(Minv_xx,fx,fy,width);
        len_Myy =  bicubiclensKernel(Minv_yy,fx,fy,width);
        len_Mxy =  bicubiclensKernel(Minv_xy,fx,fy,width);
        len_Myx =  bicubiclensKernel(Minv_xy,fx,fy,width);
        ex[i] += len_Mxx * ex_len_dx + len_Mxy * ey_len_dy;
        ey[i] += len_Myx * ex_len_dx + len_Myy * ey_len_dy;
        }
    }
}
