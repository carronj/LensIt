! Originally from https://github.com/dhanson/quicklens
! quicklens/shts/shts.f95
! --
! this file contains Fortran code implementing the spherical harmonic
! recursion relations described in quicklens/notes/shts, and routines
! to convert between vlm harmonic coefficients and complex maps.
!
! NOTE: THESE ROUTINES ARE ONLY WRITTEN FOR SPIN s>=0

subroutine HELLOfunc
  INTEGER :: NTHREADS, TID, OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM

  ! Fork a team of threads giving them their own copies of variables
  !$omp PARALLEL PRIVATE(NTHREADS, TID)

  ! Obtain thread number
  TID = OMP_GET_THREAD_NUM()
  write(*,*) 'Hello World from thread = ', TID
  ! Only master thread does this
  IF (TID .EQ. 0) THEN
    NTHREADS = OMP_GET_NUM_THREADS()
    write(*,*) 'Number of threads = ', NTHREADS
  END IF
  ! All threads join master thread and disband
  !$omp END PARALLEL
end

subroutine glm2vtm(ntht, lmax, s, tht, glm, vtm) 
! This assume ordering (healpy) ordering g[m * (2 lmax + 1 -m)/2 + l] = glm
  integer ntht, lmax,s
  double precision tht(ntht)
  double complex, intent(in)  :: glm(0:((lmax+1)*(lmax+2)/2)-1)
  double complex, intent(out) :: vtm(ntht, -lmax:lmax)

  integer l, m, tl, ts, tm, j,id
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.d0**40
  htttht(:)    = dtan(tht(:)*0.5d0)
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1.d0/dsqrt(8.d0 * acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0
  id = 0

  zl(:) = 0.d0
  if (s .ne. 0) then 
  	 ! This modifies l
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*dsqrt(1.d0 + 0.5d0 / ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.d0*s/(tl*(tl-1.d0))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  vtm(:,0) = llm_arr_p_lm0(:)*glm(l)

  rl(:) = 0.d0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
  end do

  ! We first do m = 0 recursion for all l giving sLambda_l0 contribution.
  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vtm(:,0) = vtm(:,0) + llm_arr_p_lm0(:)*glm(tl)
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal, spow,id, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, glm, vtm)
  ! for m > 0 we first build sLambda_mm:
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           ! Eq. B
           tfac = -dsqrt( 1.d0 * (s-m+1.d0) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
            ! Eq.C
           tfac = +dsqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1.0/sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm

    ! We have computed sLambda_mm with m = -tm in llm_arr_m and + tm in llm_arr_p
    ! Now we use the Legendre recursion again to do all l >= m

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.d0
	
     id = m * (2 * lmax + 1 -m)/2
     vtm(:,+m) = llm_arr_p_lm0(:)*scal(:)*glm(id + l)
     vtm(:,-m) = llm_arr_m_lm0(:)*scal(:)*glm(id + l)

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = dsqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     ! Eq. 15 for m
     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

        ! We do not need that for spin 0 (zl = 0) :
        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)

		! Could put here symmetry relations across the equator.
        vtm(:,+m) = vtm(:,+m)+llm_arr_p_lm0(:)*scal(:)*glm(id + tl)
        vtm(:,-m) = vtm(:,-m)+llm_arr_m_lm0(:)*scal(:)*glm(id + tl)

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine glm2vtm

subroutine vlm2vtm(ntht, lmax, s, tht, vlm, vtm)
  integer ntht, lmax, s
  double precision tht(ntht)
  double complex, intent(in)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, intent(out) :: vtm(ntht, -lmax:lmax)

  integer l, m, tl, ts, tm, j
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.d0**40
  !sfac = 1. this does not affect the poles
  htttht(:)    = dtan(tht(:)*0.5d0)
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1.d0/dsqrt(8.d0 * acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0

  zl(:) = 0.d0
  if (s .ne. 0) then
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*dsqrt(1.d0 + 0.5d0 / ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.d0*s/(tl*(tl-1.d0))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  vtm(:,0) = llm_arr_p_lm0(:)*vlm(l*l+l)

  rl(:) = 0.d0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
  end do

  ! We first do m = 0 recursion for all l giving sLambda_l0 contribution.
  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vtm(:,0) = vtm(:,0) + llm_arr_p_lm0(:)*vlm(tl*tl+tl)
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, vlm, vtm)
  ! for m > 0 we first build sLambda_mm:
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           ! Eq. B
           tfac = -dsqrt( 1.d0 * (s-m+1.d0) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
            ! Eq.C
           tfac = +dsqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1.0/sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm

    ! We have computed sLambda_mm with m = -tm in llm_arr_m and + tm in llm_arr_p
    ! Now we use the Legendre recursion again to do all l >= m

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.d0

     vtm(:,+m) = llm_arr_p_lm0(:)*scal(:)*vlm(l*l+l+m)
     vtm(:,-m) = llm_arr_m_lm0(:)*scal(:)*vlm(l*l+l-m)

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = dsqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     ! Eq. 15 for m
     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

        ! We do not need that for spin 0 (zl = 0) :
        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)

		! Could put here symmetry relations across the equator.
        vtm(:,+m) = vtm(:,+m)+llm_arr_p_lm0(:)*scal(:)*vlm(tl*tl+tl+m)
        vtm(:,-m) = vtm(:,-m)+llm_arr_m_lm0(:)*scal(:)*vlm(tl*tl+tl-m)

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vlm2vtm


subroutine vlm2vtm_sym(ntht, lmax, s, tht, vlm, vtm)
! Same as above but produces for all tht the result for 1 - tht as well.
! Using sLlm(pi - tht) = (-1) ** (l + s) sLl-m(tht)
  integer ntht, lmax, s
  double precision tht(ntht)
  double complex, intent(in)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, intent(out) :: vtm(0:2 * ntht -1, -lmax:lmax)

  integer l, m, tl, ts, tm, j,sgn
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.d0**40
  !sfac = 1. this does not affect the poles
  htttht(:)    = dtan(tht(:)*0.5d0)
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1.d0/dsqrt(8.d0 * acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0

  zl(:) = 0.d0
  if (s .ne. 0) then
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*dsqrt(1.d0 + 0.5d0 / ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.d0*s/(tl*(tl-1.d0))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  ! sign is 1 here since l = s
  vtm(:ntht-1,0) = llm_arr_p_lm0(:)*vlm(l*l+l)
  vtm(ntht:,0) = vtm(:ntht-1,0)
  
  rl(:) = 0.d0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
  end do

  ! We first do m = 0 recursion for all l giving sLambda_l0 contribution.
  sgn = -1
  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)
     
     vtm(:ntht-1,0) = vtm(:ntht-1,0) +  llm_arr_p_lm0(:)*vlm(tl*tl+tl)
     vtm(ntht:,0) = vtm(ntht:,0) + sgn *llm_arr_p_lm0(:)*vlm(tl*tl+tl)
     sgn = -sgn
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, sgn,scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, vlm, vtm)
  ! for m > 0 we first build sLambda_mm:
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           ! Eq. B
           tfac = -dsqrt( 1.d0 * (s-m+1.d0) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
            ! Eq.C
           tfac = +dsqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1.0/sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm
    ! We have computed sLambda_mm with m = -tm in llm_arr_m and + tm in llm_arr_p
    ! Now we use the Legendre recursion again to do all l >= m

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.d0

     sgn = (-1) ** mod(l + s,2)
     vtm(:ntht-1,+m) = llm_arr_p_lm0(:)*scal(:)*vlm(l*l+l+m)
     vtm(:ntht-1,-m) = llm_arr_m_lm0(:)*scal(:)*vlm(l*l+l-m)
     vtm(ntht:,+m) = sgn * llm_arr_m_lm0(:)*scal(:)*vlm(l*l+l+m)
     vtm(ntht:,-m) = sgn * llm_arr_p_lm0(:)*scal(:)*vlm(l*l+l-m)
     

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = dsqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     ! Eq. 15 for m
     do tl=l+1,lmax
        sgn = - sgn
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)


        vtm(:ntht-1,+m) = vtm(:ntht-1,+m)+llm_arr_p_lm0(:)*scal(:)*vlm(tl*tl+tl+m)
        vtm(:ntht-1,-m) = vtm(:ntht-1,-m)+llm_arr_m_lm0(:)*scal(:)*vlm(tl*tl+tl-m)
        vtm(ntht:,+m) = vtm(ntht:,+m)+ sgn*llm_arr_m_lm0(:)*scal(:)*vlm(tl*tl+tl+m)
        vtm(ntht:,-m) = vtm(ntht:,-m)+ sgn*llm_arr_p_lm0(:)*scal(:)*vlm(tl*tl+tl-m)
  

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vlm2vtm_sym
! This comes from quicklens with differences in internal definitions that was making the ouput single precision
! it holds vt{-m} = vtm^*  
subroutine glm2vtm_s0(ntht, lmax, tht, glm, vtm)
  ! Same as above but outputs only m = 0 to lmax + 1
  ! using for spin 0 vt-m = vtm^*
  ! Only 'upper half' of vtm array is used.

  integer ntht, lmax
  double precision tht(ntht)
  double complex, intent(in)  :: glm(0:((lmax+1)*(lmax+2)/2-1))
  double complex, intent(out) :: vtm(ntht,0:lmax) 

  integer l, m, tl, tm, j,id,s
  double precision costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax)

  sfac = 2.d0**40
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1.d0/dsqrt(8.d0 * acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0
  id = 0
  s = 0

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  vtm(:,0) = llm_arr_p_lm0(:)*glm(0)

  rl(:) = 0.d0
  do tl=1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
  end do

  ! We first do m = 0 recursion for all l giving sLambda_l0 contribution.
  do tl=1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vtm(:,0) = vtm(:,0) + llm_arr_p_lm0(:)*glm(tl)
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(id,j, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p) schedule(dynamic, 1) &
!$omp shared(costht, sintht, sfac, s, zl, ntht, tht, glm, vtm)
  ! for m > 0 we first build sLambda_mm:
  ! Fork a team of threads giving them their own copies of variables

  do tm=1,lmax
     do m=m+1,tm
        tfac = +dsqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
        llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
        l = m
        do j=1,ntht
            if (abs(llm_arr_p(j)) < 1.0/sfac) then
                llm_arr_p(j) = llm_arr_p(j)*sfac
                spow_i(j) = spow_i(j)-1
            end if
        end do
     end do
     m = tm

    ! We have computed sLambda_mm with m = -tm in llm_arr_m and + tm in llm_arr_p
    ! Now we use the Legendre recursion again to do all l >= m

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     id = m * (2 * lmax + 1 -m)/2
     vtm(:,+m) = llm_arr_p_lm0(:)*scal(:)*glm(id + l)

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = dsqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     ! Eq. 15 for m
     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

		! Could put here symmetry relations across the equator.
        vtm(:,+m) = vtm(:,+m)+llm_arr_p_lm0(:)*scal(:)*glm(id + tl)

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine glm2vtm_s0

subroutine glm2vtm_s0sym(ntht, lmax, tht, glm, vtm)
  ! Same as above but outputs only m = 0 to lmax + 1
  ! using for spin 0 vt-m = vtm^*
  ! for each tht, pi -tht is also calculated using Llm(t) = (-)^(l+m)Llm(pi -t)
  ! Not certain it works for tht = 0.
  ! Only 'upper half' of vtm array is used.
  integer ntht, lmax,sgn
  double precision tht(ntht)
  double complex, intent(in)  :: glm(0:((lmax+1)*(lmax+2)/2-1))
  double complex, intent(out) :: vtm(0:2 * ntht-1,0:lmax)

  integer l, m, tl, tm, j,id,s
  double precision costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax)
  double complex llm_cont(ntht)

  sfac = 2.d0**40
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1.d0/dsqrt(8.d0 * acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  s = 0
  m = 0
  l = 0
  id = 0
  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  vtm(0:ntht-1,0) = llm_arr_p_lm0(:)*glm(l*l+l)
  vtm(ntht:,0) = vtm(0:ntht-1,0)
  
  
  rl(:) = 0.d0
  do tl=1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
  end do

  ! We first do m = 0 recursion for all l giving sLambda_l0 contribution.
  sgn = -1
  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)
     
     llm_cont(:) = llm_arr_p_lm0(:)*glm(tl)
     vtm(0:ntht-1,0) = vtm(0:ntht-1,0) + llm_cont(:)
     vtm(ntht:,0) = vtm(ntht:,0) + sgn * llm_cont(:)
     sgn = -sgn
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j,id,sgn, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_x_lmt,llm_cont,rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p) schedule(dynamic, 1) &
!$omp shared(costht, sintht, sfac, s, zl, ntht, tht, glm, vtm)
  ! for m > 0 we first build sLambda_mm:
  do tm=1,lmax
     do m=m+1,tm
        tfac = +dsqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
        llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
        l = m
        do j=1,ntht
            if (abs(llm_arr_p(j)) < 1.0/sfac) then
                llm_arr_p(j) = llm_arr_p(j)*sfac
                spow_i(j) = spow_i(j)-1
            end if
        end do
     end do
     m = tm

    ! We have computed sLambda_mm with m = -tm in llm_arr_m and + tm in llm_arr_p
    ! Now we use the Legendre recursion again to do all l >= m

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     id = m * (2 * lmax + 1 -m)/2
     vtm(0:ntht-1,m)  = llm_arr_p_lm0(:)*scal(:)*glm(id + l)
     vtm(ntht:,m) = vtm(0:ntht-1,m)

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = dsqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     ! Eq. 15 for m
     sgn = -1
     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

		! Could put here symmetry relations across the equator.
        llm_cont(:) = llm_arr_p_lm0(:)*scal(:)*glm(id + tl)
        vtm(0:ntht-1,m) = vtm(0:ntht-1,m) + llm_cont(:)
        vtm(ntht:,m) = vtm(ntht:,m) + sgn * llm_cont(:)
        sgn = -sgn 

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine glm2vtm_s0sym

subroutine map2vlm(ntht, nphi, lmax, s, tht, phi, map, vlm)
  integer ntht, nphi, lmax, s
  double precision tht(ntht), phi(nphi)
  double complex, intent(in), dimension(ntht, nphi) :: map
  double complex, intent(out)  :: vlm(0:((lmax+1)*(lmax+1)-1))
  double complex, save, allocatable ::  vtm(:,:)
  !f2py intent(in)   :: ntht, nphi, s, lmax
  !f2py intent(in)   :: tht, phi, map
  !f2py intent(hide) :: ntht, nphi
  !f2py intent(out)  :: vlm

  allocate( vtm(ntht,-lmax:lmax) )
  vtm(:,:) = 0.0
  vlm(:)   = 0.0

  call map2vtm(ntht, nphi, lmax, phi, map, vtm)
  call vtm2vlm(ntht, lmax, s, tht, vtm, vlm)
  
  deallocate(vtm)
end subroutine map2vlm

subroutine map2vtm(ntht, nphi, lmax, phi, map, vtm)
  integer ntht, nphi, lmax
  double precision phi(nphi)
  double complex, intent(in)  :: map(ntht, nphi)
  double complex, intent(out) :: vtm(ntht, -lmax:lmax)

  integer p, m

  do p=1,nphi
     vtm(:,0) = vtm(:,0) + map(:,p)
  end do

!$omp parallel do default(shared)
  do p=1,nphi
     do m=1,lmax
        vtm(:,+m) = vtm(:,+m) + map(:,p) * &
             (cos(phi(p)*m)-(0.0,1.0)*sin(phi(p)*m))
        vtm(:,-m) = vtm(:,-m) + map(:,p) * &
             (cos(phi(p)*m)+(0.0,1.0)*sin(phi(p)*m))
     end do
  end do
!$omp end parallel do
end subroutine map2vtm

subroutine vtm2vlm(ntht, lmax, s, tht, vtm, vlm)
! This calculates sum_th sLlm(tht) vtm(theta,m)
! vtm is meant to be e.g. sum_l slm(tht) vlm
! such that vtm2vlm(sin(tht)*(2. * np.pi) * vtm) should give vlm for fine enough grids.
! could accelerate that by assuming input is symm in tht. (hmm would only save memory)
  integer ntht, lmax, s
  double precision tht(ntht)
  double complex, intent(in)  :: vtm(ntht, -lmax:lmax)
  double complex, intent(out) :: vlm(0:((lmax+1)*(lmax+1)-1))

  integer l, m, tl, ts, tm, j
  double precision htttht(ntht), costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht), llm_arr_m(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_m_lm0(ntht), llm_arr_m_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.**40

  htttht(:)    = dtan(tht(:)*0.5d0)
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1./dsqrt(8.d0*acos(0.d0))
  llm_arr_m(:) = llm_arr_p(:)

  l = 0
  m = 0

  zl(:) = 0.d0
  if (s .ne. 0) then
     do ts=1,s
        llm_arr_p(:) = llm_arr_p(:)*dsqrt(1.d0+0.5d0/ts)*sintht(:)
     end do
     llm_arr_m(:) = llm_arr_p(:)
     l = s

     do tl=2,lmax
        zl(tl) = 1.d0*s/(tl*(tl-1.))
     end do
  end if

  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  vlm(l*l+l) = sum( llm_arr_p_lm0(:)*vtm(:,0) )

  rl(:) = 0.d0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl - s*s) / (tl*tl * (4.d0*tl*tl-1.d0)) )
  end do

  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl) 

     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)

     vlm(tl*tl+tl) = sum( llm_arr_p_lm0(:)*vtm(:,0) )
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_m_lm0, llm_arr_m_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p, llm_arr_m) schedule(dynamic, 1) &
!$omp shared(htttht, costht, sintht, sfac, s, zl, ntht, tht, vlm, vtm)
  do tm=1,lmax
     do m=m+1,tm
        if (m<=s) then
           tfac = -sqrt( 1.d0 * (s-m+1.d0) / (s+m) )
           llm_arr_p(:) = +llm_arr_p(:) * htttht(:) * tfac
           llm_arr_m(:) = +llm_arr_m(:) / htttht(:) * tfac
           l = s
        else
           tfac = +sqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m+s)*(m-s)) )
           llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
           llm_arr_m(:) = +llm_arr_m(:) * tfac * sintht(:)
           l = m

           do j=1,ntht
              if (abs(llm_arr_p(j)) < 1./sfac) then
                 llm_arr_p(j) = llm_arr_p(j)*sfac
                 llm_arr_m(j) = llm_arr_m(j)*sfac
                 spow_i(j) = spow_i(j)-1
              end if
           end do
        end if
     end do
     m = tm

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     llm_arr_m_lm0(:) = llm_arr_m(:)
     llm_arr_m_lm1(:) = 0.d0

     vlm(l*l+l+m) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(:,+m) )
     vlm(l*l+l-m) = sum( llm_arr_m_lm0(:)*scal(:)*vtm(:,-m) )

     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = sqrt( 1.d0 * (tl*tl - m*m) * (tl*tl - s*s) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     do tl=l+1,lmax
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)
		! Can simplify this for spin 0 : 0Ll-m = (-1)^m 0Llm
        llm_arr_x_lmt(:) = (llm_arr_m_lm0(:) * (costht(:) - m*zl(tl)) - llm_arr_m_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_m_lm1(:) = llm_arr_m_lm0(:)
        llm_arr_m_lm0(:) = llm_arr_x_lmt(:)

        vlm(tl*tl+tl+m) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(:,+m) )
        vlm(tl*tl+tl-m) = sum( llm_arr_m_lm0(:)*scal(:)*vtm(:,-m) )

        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 llm_arr_m_lm0(j) = llm_arr_m_lm0(j)/sfac
                 llm_arr_m_lm1(j) = llm_arr_m_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vtm2vlm

subroutine vtm2alm_syms0(ntht, lmax, tht, vtm,alm)
! This calculates sum_th sLlm(tht) vtm(theta,m)
! vtm is meant to be e.g. sum_l slm(tht) vlm
! such that vtm2vlm(sin(tht)*(2. * np.pi) * vtm)
! this uses symmetry tricks
! using for spin 0 vt-m = vtm^*
! Could further reduce that by a factor of two
  integer ntht, lmax
  double precision tht(ntht)
! 0:nth :symmetric part, nth: antysmetric part
  double complex, intent(in)  :: vtm(0:2 * ntht-1,0:lmax)
  double complex, intent(out) :: alm(0:((lmax+1)*(lmax+2)) / 2-1)

  integer l, m, tl, tm, j,parity
  double precision costht(ntht), sintht(ntht)
  double precision scal(ntht), spow(ntht), spow_i(ntht)

  double precision tfac, sfac, llm_arr_p(ntht)
  double precision llm_arr_p_lm0(ntht), llm_arr_p_lm1(ntht)
  double precision llm_arr_x_lmt(ntht), rl(0:lmax), zl(0:lmax)

  sfac = 2.**40
  costht(:)    = dcos(tht(:))
  sintht(:)    = dsin(tht(:))

  llm_arr_p(:) = 1./dsqrt(8.d0*acos(0.d0))

  l = 0
  m = 0

  zl(:) = 0.d0
  ! do m=0
  llm_arr_p_lm0(:) = llm_arr_p(:)
  llm_arr_p_lm1(:) = 0.d0

  parity = mod(l - m,2)
  alm(m * (2 * lmax + 1 -m)/2 + l) = sum( llm_arr_p_lm0(:)*vtm(ntht * parity:(1 + parity) * ntht - 1,0) )

  rl(:) = 0.d0
  do tl=l+1,lmax
     rl(tl) = sqrt( 1.d0 * tl*tl * (tl*tl) / (tl*tl * (4.d0*tl*tl-1.d0)) )
  end do
  do tl=l+1,lmax
     llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * costht(:) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
     llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
     llm_arr_p_lm0(:) = llm_arr_x_lmt(:)
     parity = mod(tl,2)
     alm(m * (2 * lmax + 1 -m)/2 + tl) = sum( llm_arr_p_lm0(:)*vtm(ntht * parity:(1 + parity) * ntht - 1,0) )
  end do

  spow_i(:) = 0.d0

!$omp parallel do default(none) &
!$omp private(j, tl, tm, scal,parity, spow, tfac, llm_arr_p_lm0) &
!$omp private(llm_arr_p_lm1, llm_arr_x_lmt, rl) &
!$omp firstprivate(l, m, lmax, spow_i, llm_arr_p) schedule(dynamic, 1) &
!$omp shared(costht, sintht, sfac, zl, ntht, tht, vlm, vtm, alm)
  do tm=1,lmax ! m loop
     do m=m+1,tm
        tfac = +sqrt( 1.d0 * m * (2.d0*m+1.d0)/(2.d0*(m)*(m)) )
        llm_arr_p(:) = -llm_arr_p(:) * tfac * sintht(:)
        l = m
        do j=1,ntht
           if (abs(llm_arr_p(j)) < 1./sfac) then
              llm_arr_p(j) = llm_arr_p(j)*sfac
              spow_i(j) = spow_i(j)-1
           end if
        end do
     end do
     m = tm

     spow(:) = spow_i(:)
     scal(:) = sfac**(spow(:))

     llm_arr_p_lm0(:) = llm_arr_p(:)
     llm_arr_p_lm1(:) = 0.d0

     parity = mod(l - m,2)
     alm(m * (2 * lmax + 1 -m)/2 + l) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(ntht * parity:(1 + parity) * ntht - 1,m) )
     rl(:) = 0.d0
     do tl=l+1,lmax
        rl(tl) = sqrt( 1.d0 * (tl*tl - m*m) * (tl*tl) / (tl*tl * (4.*tl*tl-1.d0)) )
     end do

     do tl=l+1,lmax ! l loop starting from m + 1
        llm_arr_x_lmt(:) = (llm_arr_p_lm0(:) * (costht(:) + m*zl(tl)) - llm_arr_p_lm1(:) * rl(tl-1)) / rl(tl)
        llm_arr_p_lm1(:) = llm_arr_p_lm0(:)
        llm_arr_p_lm0(:) = llm_arr_x_lmt(:)
        parity = mod(tl - m,2)
        alm(m * (2 * lmax + 1 -m)/2 + tl) = sum( llm_arr_p_lm0(:)*scal(:)*vtm(ntht * parity:(1 + parity) * ntht - 1,m) )
        if (mod(tl,10) == 0) then
           do j=1,ntht
              if (abs(llm_arr_p_lm0(j)) > sfac) then
                 llm_arr_p_lm0(j) = llm_arr_p_lm0(j)/sfac
                 llm_arr_p_lm1(j) = llm_arr_p_lm1(j)/sfac
                 spow(j) = spow(j) + 1
                 scal(j) = sfac**(spow(j))
              end if
           end do
        end if
     end do
  end do
!$omp end parallel do
end subroutine vtm2alm_syms0