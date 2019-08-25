from __future__ import print_function

import numpy as np


def IsPowerOfTwo(i):
    """Returns true if all entries of i are powers of two, False otherwise.

    """
    return (i & (i - 1)) == 0 and i != 0


def Log2ofPowerof2(shape):
    """ Returns powers of two exponent for each element of shape

    """
    res = np.array(shape)
    for i in range(res.size):
        n = shape[i]
        assert (IsPowerOfTwo(n)), "Invalid input"
        ix = 0
        while n > 1:
            n //= 2
            ix += 1
        res[i] = ix
    return res


def Freq(i, N):
    """Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]

     in numpy fft convention as integer i runs from 0 to N-1.
     Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
     Both inputs must be integers.
     All entries of N must be even.


    """
    assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
    return i - 2 * (i >= (N / 2)) * (i % (N / 2))


def rfft2_reals(shape):
    """Pure reals modes in 2d rfft array. (from real map shape, not rfft array)

    """
    N0, N1 = shape
    fx = [0]
    fy = [0]
    if N1 % 2 == 0: fx.append(0); fy.append(N1 // 2)
    if N0 % 2 == 0: fx.append(N0 / 2); fy.append(0)
    if N1 % 2 == 0 and N0 % 2 == 0: fx.append(N0 // 2); fy.append(N1 // 2)
    return np.array(fx), np.array(fy)


def upgrade_map(LD_map, HD_res):
    """
    Upgrade LD_map to a higher resolution map, using rfft and back.
    :param LD_map: Must have shape entries powers of two.
    :param HD_res:
    :return: Same map at higher resolution.
    """
    LD_res = Log2ofPowerof2(LD_map.shape)
    if np.all(LD_res == HD_res): return LD_map
    assert np.all(HD_res >= LD_res)
    HD_rshape = (2 ** HD_res[0], 2 ** (HD_res[1] - 1) + 1)
    HD_shape = (2 ** HD_res[0], 2 ** HD_res[1])

    rfft = np.fft.rfft2(LD_map)
    ret_rfft = np.zeros(HD_rshape, dtype=complex)
    ret_rfft[0:rfft.shape[0] / 2 + 1, 0:rfft.shape[1]] = rfft[0:rfft.shape[0] / 2 + 1, :]  # positive frequencies
    # negative frequencies :
    ret_rfft[HD_rshape[0] - rfft.shape[0] + rfft.shape[0] / 2:, 0:rfft.shape[1]] = rfft[rfft.shape[0] / 2:, :]
    fac_LDrfft2HDrfft = 2 ** (HD_res[0] - LD_res[0] + HD_res[1] - LD_res[1])
    return np.fft.irfft2(ret_rfft, HD_shape) * fac_LDrfft2HDrfft


def subsample(HD_map, LD_res):
    """Simple subsampling of map.

    """
    HD_res = Log2ofPowerof2(HD_map.shape)
    if np.all(LD_res == HD_res): return HD_map.copy()
    assert np.all(HD_res >= LD_res)
    return HD_map[0::2 ** (HD_res[0] - LD_res[0]), 0::2 ** (HD_res[1] - LD_res[1])]


def supersample(LD_map, HD_shape):
    """Simple hypersampling of map.

    """
    if LD_map.shape == HD_shape: return LD_map.copy()
    assert np.all(np.array(HD_shape) > np.array(LD_map.shape))
    assert np.all(np.array(HD_shape) % np.array(LD_map.shape) == 0.)
    HDmap = np.zeros(HD_shape)
    fac0, fac1 = (HD_shape[0] / LD_map.shape[0], HD_shape[1] / LD_map.shape[1])
    for i in range(fac0):
        for j in range(fac1):
            HDmap[i::fac0, j::fac1] = LD_map
    return HDmap


def degrade(HD_map, LD_shape):
    if np.all(HD_map.shape <= LD_shape): return HD_map.copy()
    fac0, fac1 = (HD_map.shape[0] / LD_shape[0], HD_map.shape[1] / LD_shape[1])
    assert fac0 * LD_shape[0] == HD_map.shape[0] and fac1 * LD_shape[1] == HD_map.shape[1], (
        (fac0, fac1), LD_shape, HD_map.shape)

    ret = np.zeros(LD_shape, dtype=HD_map.dtype)
    for _i in range(fac0):
        sl0 = slice(_i, HD_map.shape[0], fac0)
        for _j in range(fac1):
            sl1 = slice(_j, HD_map.shape[1], fac1)
            ret += HD_map[sl0, sl1]
    return ret * (1. / (fac0 * fac1))


def degrade_mask(mask, LD_shape):
    # FIXME :
    dmask = degrade(mask, LD_shape)
    return dmask  # * (dmask >= 1.)


def udgrade_rfft2(rfft2map, shape, norm=False):
    assert norm == False, 'not implemented'  # norm. factor for rfft normalization maps. Here just shuffling indices.
    if shape == (rfft2map.shape[0], 2 * (rfft2map.shape[1] - 1)):
        return rfft2map
    assert np.all([s % 2 == 0 for s in shape]), shape
    assert rfft2map.shape[0] % 2 == 0
    rshape = np.array((shape[0], shape[1] / 2 + 1))
    if np.all(np.array(rshape) >= rfft2map.shape):
        return _upgrade_rfft2(rfft2map, shape)
    elif np.all(np.array(rshape) <= rfft2map.shape):
        return _degrade_rfft2(rfft2map, shape)
    else:
        assert 0, 'not implemented'


def _degrade_rfft2(rfft2map, LDshape):
    ret = np.zeros((LDshape[0], LDshape[0] / 2 + 1), dtype=complex)
    ret[0:LDshape[0] / 2 + 1, :] = rfft2map[0:LDshape[0] / 2 + 1, 0:ret.shape[1]]
    ret[LDshape[0] / 2::] = rfft2map[rfft2map.shape[0] - LDshape[0] / 2:, 0:ret.shape[1]]
    # Corrections for pure reals and (-k) = k* :
    ret[LDshape[0] / 2 + 1:, -1] = ret[1:LDshape[0] / 2, -1][::-1].conj()
    ret[rfft2_reals(LDshape)].imag = 0.
    return ret


def _upgrade_rfft2(rfft2map, HDshape):
    ret = np.zeros((HDshape[0], HDshape[0] / 2 + 1), dtype=complex)
    # positive 0axis frequencies : (including N/2 + 1, which is pure real.
    ret[0:rfft2map.shape[0] / 2 + 1, 0:rfft2map.shape[1]] = rfft2map[0:rfft2map.shape[0] / 2 + 1, 0:rfft2map.shape[1]]
    # Negative 0axis freq.
    ret[HDshape[0] - rfft2map.shape[0] / 2:HDshape[0], 0:rfft2map.shape[1]] \
        = rfft2map[rfft2map.shape[0] / 2:, 0:rfft2map.shape[1]]
    return ret
