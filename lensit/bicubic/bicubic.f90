double precision function cubicfilter(x, c0, c1, c2, c3)
    ! filter 4 values using cubic splines
    double precision x, c0, c1, c2, c3
    double precision w0, w1, w2, w3
    w0 = x*(x*(-x + 3d0) - 3d0) + 1d0
    w1 = x*x*(3d0*x - 6d0) + 4d0;
    w2 =  x*(x*(-3d0*x + 3d0) + 3d0) + 1d0
    w3 =  x*x*x
    cubicfilter = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) * 0.16666666666666666d0
end

double precision function tex2d(ftl_map, x, y, nx, ny)
    ! nx : second dimension of array
    ! ny : first dimension of array
    implicit none
    double precision, intent(in) :: ftl_map(0:ny-1, 0:nx-1)
    integer, intent(in) :: nx, ny
    integer x, y

    if (y < 0)  y = mod(ny + y, ny)
    if (x < 0)  x = mod(nx + x, nx)
    if (y >= ny) y = mod(y, ny)
    if (x >= nx) x = mod(x, nx)
    tex2d = ftl_map(y, x)
end

double precision function eval(ftl_map, fx, fy, nx, ny)
    ! nx : second dimension of array
    ! ny : first dimension of array
    implicit none
    double precision, intent(in) :: ftl_map(0:ny-1, 0:nx-1), fx, fy
    integer, intent(in) :: nx, ny
    double precision, external :: cubicfilter, tex2d
    double precision gx, gy
    integer px, py

    px = floor(fx)
    py = floor(fy)
    gx = fx - px
    gy = fy - py
    eval = cubicfilter(gy, &
              cubicfilter(gx, tex2d(ftl_map, px-1, py-1,nx, ny), tex2d(ftl_map, px, py-1,nx, ny), &
                        tex2d(ftl_map, px+1, py-1,nx, ny), tex2d(ftl_map, px+2,py-1,nx, ny)), &
              cubicfilter(gx, tex2D(ftl_map, px-1, py, nx, ny),   tex2d(ftl_map, px, py,nx, ny),   &
                        tex2d(ftl_map, px+1, py, nx, ny),   tex2d(ftl_map, px+2, py,nx, ny)),&
              cubicfilter(gx, tex2D(ftl_map, px-1, py+1,nx, ny), tex2d(ftl_map, px, py+1,nx, ny), &
                        tex2d(ftl_map, px+1, py+1, nx, ny), tex2d(ftl_map, px+2, py+1,nx, ny)), &
              cubicfilter(gx, tex2D(ftl_map, px-1, py+2,nx, ny), tex2d(ftl_map, px, py+2,nx, ny), &
                        tex2d(ftl_map, px+1, py+2,nx, ny), tex2d(ftl_map, px+2, py+2,nx, ny)) )
end

subroutine deflect(output, ftl_map, fx, fy, nx, ny, npts)
    ! input ftl_map should be bicubic prefiltered map
    ! fx, fy new coordinate in grid units.
    ! x and y are the 2nd and 1st ftl_map array dimension respectively.
    ! fx new x-coordinate, fy new y-coordinate

    implicit none
    double precision, intent(in) :: ftl_map(0:ny-1,0:nx-1)
    double precision, intent(in) :: fx(0:npts-1), fy(0:npts-1)
    double precision, intent(out) :: output(0:npts-1)
    double precision, external :: eval
    integer, intent(in) :: nx, ny, npts
    integer i
    do i = 0, npts - 1
        output(i) = eval(ftl_map, fx(i), fy(i), nx, ny)
    end do
end subroutine deflect

subroutine deflect_inverse(exo, eyo, ex, ey, dx, dy, minv_xx, minv_yy, minv_xy, minv_yx, nx, ny)
    ! iterate deflection inversion estimate
    ! ex, ey are current estimates in grid units, dx, dy the deflection in grid units
    ! minv_XX are the magnification matrix inverse components
    ! exo, eyo are the improved estimates.

    implicit none
    double precision, intent(in) :: dx(0:ny-1, 0:nx-1), dy(0:ny-1, 0:nx-1)
    double precision, intent(in) :: minv_xx(0:ny-1, 0:nx-1), minv_yy(0:ny-1, 0:nx-1)
    double precision, intent(in) :: minv_xy(0:ny-1, 0:nx-1), minv_yx(0:ny-1, 0:nx-1)
    double precision, intent(in) :: ex(0:ny-1, 0:nx-1), ey(0:ny-1, 0:nx-1)
    double precision, intent(out) :: exo(0:ny-1, 0:nx-1), eyo(0:ny-1, 0:nx-1)

    double precision fx, fy
    double precision len_mxx, len_myy, len_mxy, len_myx
    double precision ex_len_dx, ey_len_dy
    double precision, external :: eval
    integer x, y, nx, ny

    do x = 0, nx -1
        do y = 0, ny -1
            fx = ex(y, x) + x
            fy = ey(y, x) + y
            ex_len_dx = ex(y, x) + eval(dx, fx, fy, nx, ny)
            ey_len_dy = ey(y, x) + eval(dy, fx, fy, nx, ny)
            len_mxx = eval(minv_xx, fx, fy, nx, ny)
            len_myy = eval(minv_yy, fx, fy, nx, ny)
            len_mxy = eval(minv_xy, fx, fy, nx, ny)
            len_myx = eval(minv_yx, fx, fy, nx, ny)
            exo(y, x) = ex(y, x) + len_mxx * ex_len_dx + len_mxy * ey_len_dy
            eyo(y, x) = ey(y, x) + len_myx * ex_len_dx + len_myy * ey_len_dy
        end do
    end do

end subroutine deflect_inverse

