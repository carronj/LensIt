import numpy as np


class periodicmap_spliter:
    def __init__(self):
        pass

    @staticmethod
    def get_slices_chk_N(N, LD_res, HD_res, buffers, inverse=False):
        """

        This lengthy, irritating piece of code returns the slice idcs for subcube (i,j)
        in a decomposition of an original map of shape (2**HD_res[0],2**HD_res[1]) with chunks of 2**LD_res per sides,
        together with buffers[0],[1] buffers pixel on each side, fixed by the periodicity condition of the HD map.
        Nothing very subtle for the map interior, but the edges and corners require a little bit of care.

        E.g. this is how this can be used to extract a LD cube:
        cub = np.zeros((2**LD_res[0]+ 2*buffers[0],2**LD_res[1] + 2*buffers[1]))
        sLDs,sHDs = spliter.get_slices_chk_N(N,LD_res,HD_res,buffers)
        for sLD,sHD in zip(sLDs,sHDs): cub[sLD] =  map[sHD].

        Similary to patch the pieces together to a big cube. In that case you need only the first elements
        of the sLDs, sHDs arrays. For instance

        newmap = np.zeros((2**HD_res[0],2**HD_res[1]))
        for i in xrange(0,2 ** (HD_res[0] - LD_res[0])):
            for j in xrange(0,2 ** (HD_res[1] - LD_res[1])):
                sLDs,sHDs = spliter.get_slices_chk_ij(i,j,LD_res,HD_res,buffers,inverse = True)
                newmap[sHDs[0]] =  cubes[i*2 ** (HD_res[1] - LD_res[1]) + j][sLDs[0]]

        does patch together the cubes sequence to build the bigger cube.

        """
        assert len(LD_res) == 2 and len(HD_res) == 2
        if np.all(LD_res == HD_res):
            assert N == 0, N
            assert buffers == (0, 0), buffers
            sl0_LD = slice(0, 2 ** LD_res[0])  # Center of buffered cube
            sl1_LD = slice(0, 2 ** LD_res[1])
            sl0_HD = slice(0, 2 ** HD_res[0])  # Center of buffered cube
            sl1_HD = slice(0, 2 ** HD_res[1])
            ret_LD = [(sl0_LD, sl1_LD)]
            ret_HD = [(sl0_HD, sl1_HD)]
            return ret_LD, ret_HD

        assert np.all(LD_res < HD_res)
        assert len(buffers) == 2
        assert buffers[0] < 2 ** LD_res[0] and buffers[1] < 2 ** LD_res[1]

        N0 = 2 ** LD_res[0]  # shape of small cube, buffers excl.
        N1 = 2 ** LD_res[1]
        N0H = 2 ** HD_res[0]  # shape of large cube
        N1H = 2 ** HD_res[1]

        Nchks_0 = 2 ** (HD_res[0] - LD_res[0])
        Nchks_1 = 2 ** (HD_res[1] - LD_res[1])
        assert N < Nchks_1 * Nchks_0, N

        b0 = buffers[0]
        b1 = buffers[1]
        ret_LD = []
        ret_HD = []
        j = N % Nchks_1
        i = N // Nchks_1  # in 0, ..., Nchks_0 -1

        if inverse:
            # We want the inverse mapping only
            sl0_LD = slice(b0, N0 + b0)  # Center of buffered cube
            sl1_LD = slice(b1, N1 + b1)
            sl0_HD = slice(i * N0, (i + 1) * N0)  # slices of HD cube
            sl1_HD = slice(j * N1, (j + 1) * N1)  # slices of HD cube
            ret_LD.append((sl0_LD, sl1_LD))
            ret_HD.append((sl0_HD, sl1_HD))
            return ret_LD, ret_HD

        if 0 < i < Nchks_0 - 1:
            # i in the interior :
            sl0_LD = slice(0, N0 + 2 * b0)  # Slices of LD cube
            sl0_HD = slice(i * N0 - b0, (i + 1) * N0 + b0)  # slices of HD cube
            if 0 < j < Nchks_1 - 1:
                # We are in the interior, no big deal
                sl1_LD = slice(0, N1 + 2 * b1)
                sl1_HD = slice(j * N1 - b1, (j + 1) * N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
            elif j == 0:
                sl1_LD = slice(b1, N1 + 2 * b1)
                sl1_HD = slice(0, N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl1_LD = slice(0, b1)
                sl1_HD = slice(2 ** HD_res[1] - b1, 2 ** HD_res[1])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
            else:
                assert j == Nchks_1 - 1
                sl1_LD = slice(0, N1 + b1)
                sl1_HD = slice(2 ** HD_res[1] - N1 - b1, 2 ** HD_res[1])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl1_LD = slice(N1 + b1, N1 + 2 * b1)
                sl1_HD = slice(0, b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
        elif i == 0:
            # Bulk 0 slices
            sl0_LD = slice(b0, N0 + 2 * b0)
            sl0_HD = slice(0, N0 + b0)

            if j == 0:
                # Upper left corner. Two tweaks.
                # Bulk :
                sl1_LD = slice(b1, N1 + 2 * b1)
                sl1_HD = slice(0, N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                sl1_LD = slice(b1, N1 + 2 * b1)
                sl1_HD = slice(0, N1 + b1)
                sl0_LD = slice(0, b0)
                sl0_HD = slice(N0H - b0, N0H)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(b0, N0 + 2 * b0)
                sl0_HD = slice(0, N0 + b0)
                sl1_LD = slice(0, b1)
                sl1_HD = slice(N1H - b1, N1H)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(0, b0)
                sl1_LD = slice(0, b1)
                sl0_HD = slice(N0H - b0, N0H)
                sl1_HD = slice(N1H - b1, N1H)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD

            elif j == Nchks_1 - 1:
                # upper right corner
                # Bulk :
                sl1_LD = slice(0, N1 + b1)
                sl1_HD = slice(2 ** HD_res[1] - N1 - b1, 2 ** HD_res[1])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(0, b0)
                sl0_HD = slice(N0H - b0, N0H)
                sl1_LD = slice(0, N1 + b1)
                sl1_HD = slice(N1H - N1 - b1, N1H)

                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl1_LD = slice(N1 + b1, N1 + 2 * b1)
                sl1_HD = slice(0, b1)
                sl0_LD = slice(b0, N0 + 2 * b0)
                sl0_HD = slice(0, b0 + N0)

                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                # Last little square is missing :
                sl0_LD = slice(0, b0)
                sl0_HD = slice(N0H - b0, N0H)
                sl1_LD = slice(N1 + b1, N1 + 2 * b1)
                sl1_HD = slice(0, b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
            else:
                assert 0 < j < Nchks_1 - 1
                sl1_LD = slice(0, N1 + 2 * b1)
                sl1_HD = slice(j * N1 - b1, (j + 1) * N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(0, b0)
                sl0_HD = slice(2 ** HD_res[0] - b0, 2 ** HD_res[0])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD

        elif i == Nchks_0 - 1:
            sl0_LD = slice(0, N0 + b0)
            sl0_HD = slice(2 ** HD_res[0] - N0 - b0, 2 ** HD_res[0])
            if j == 0:
                # lower left corner. Two tweaks.
                # Bulk :
                sl1_LD = slice(b1, N1 + 2 * b1)
                sl1_HD = slice(0, N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl1_LD = slice(0, b1)
                sl1_HD = slice(2 ** HD_res[1] - b1, 2 ** HD_res[1])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl0_HD = slice(0, b0)
                sl1_LD = slice(b1, N1 + 2 * b1)
                sl1_HD = slice(0, N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl1_LD = slice(0, b1)
                sl0_HD = slice(0, b0)
                sl1_HD = slice(N1H - b1, N1H)

                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD

            elif j == Nchks_1 - 1:
                # Lower right corner
                # Bulk :
                sl1_LD = slice(0, N1 + b1)
                sl1_HD = slice(2 ** HD_res[1] - N1 - b1, 2 ** HD_res[1])
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl1_LD = slice(N1 + b1, N1 + 2 * b1)
                sl1_HD = slice(0, b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl0_HD = slice(0, b0)
                sl1_LD = slice(0, N1 + b1)
                sl1_HD = slice(N1H - N1 - b1, N1H)

                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl1_LD = slice(N1 + b1, N1 + 2 * b1)
                sl0_HD = slice(0, b0)
                sl1_HD = slice(0, b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD

            else:
                assert 0 < j < Nchks_1 - 1
                sl1_LD = slice(0, N1 + 2 * b1)
                sl1_HD = slice(j * N1 - b1, (j + 1) * N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl0_HD = slice(0, b0)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
