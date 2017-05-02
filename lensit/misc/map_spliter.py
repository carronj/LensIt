import numpy as np


class periodicmap_spliter():
    def __init__(self):
        pass

    def splitin4(self, map, buffers):
        """
        Outputs a list of 4 maps with shape 1/2 of the input map  + 2*buffers,
        with the four corners of the input map at the center of the 4 chunks.
        The buffers are filled in assuming periodicity of the maps.
        """
        return self._mk_4chunks_by_rolling(map, buffers)

    def get_splitin4_chunk_i(self, map, buffers, i):
        assert i >= 0 and i < 4, i
        return self._mk_4chunks_by_rolling(map, buffers, i=i)[0]

    def splitin4_inverse(self, chks, buffers):
        """
        Inverse operation to self.splitin4.
        Takes list of 4 chunks and buffer sizes as input
        """
        assert len(chks) == 4, "Weird input, should be the output of splitin4"
        assert len(buffers) == 2
        N0 = (chks[0].shape[0] - 2 * buffers[0])
        N1 = (chks[0].shape[1] - 2 * buffers[1])
        map = np.zeros((N0 * 2, N1 * 2))
        sl0 = slice(buffers[0], N0 + buffers[0])
        sl1 = slice(buffers[1], N1 + buffers[1])
        map[0:N0, 0:N1] = chks[0][sl0, sl1]
        map[N0:, 0:N1] = chks[1][sl0, sl1]
        map[0:N0, N1:] = chks[2][sl0, sl1]
        map[N0:, N1:] = chks[3][sl0, sl1]
        return map

    def _split_sgns4chks(self, i):
        if i == -1:
            return [(1, 1), (-1, 1), (1, -1), (-1, -1)]  # Full set
        else:
            return [[(1, 1), (-1, 1), (1, -1), (-1, -1)][i]]

    def _mk_4chunks_by_rolling(self, map, buffers, i=-1):
        # TODO : can't be the best, really. But don't bother.
        assert len(map.shape) == 2, 'Lets keep it simple'
        assert map.shape[0] % 2 == 0 and map.shape[1] % 2 == 0, 'I said, lets keep it simple'
        assert map.shape[0] / 2 > 2 * buffers[0] and map.shape[1] / 2 > 2 * buffers[1], \
            'Buffer too large (or map too small...)'
        assert len(buffers) == 2 and type(buffers[0] == int) and type(buffers[1] == int)
        N0, N1 = map.shape
        # Chunk side lengths :
        chk_len0 = N0 / 2 + 2 * buffers[0]
        chk_len1 = N1 / 2 + 2 * buffers[1]
        chks = []
        for sgns in self._split_sgns4chks(i):
            sgn0, sgn1 = sgns
            newmap = np.roll(map, sgn0 * buffers[0], axis=0)
            newmap = np.roll(newmap, sgn1 * buffers[1], axis=1)
            s0 = (N0 - chk_len0) * abs((sgn0 - 1) / 2)
            s1 = (N1 - chk_len1) * abs((sgn1 - 1) / 2)
            chks.append(newmap[s0:s0 + chk_len0, s1:s1 + chk_len1])
        return chks

    def get_slices_chk_N(self, N, LD_res, HD_res, buffers, inverse=False):
        """
        This lengthy irritating piece of code returns the slice idcs for subcube (i,j)
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
        i = N / Nchks_1  # in 0, ..., Nchks_0 -1

        if inverse:
            # We want the inverse mapping only
            sl0_LD = slice(b0, N0 + b0)  # Center of buffered cube
            sl1_LD = slice(b1, N1 + b1)
            sl0_HD = slice(i * N0, (i + 1) * N0)  # slices of HD cube
            sl1_HD = slice(j * N1, (j + 1) * N1)  # slices of HD cube
            ret_LD.append((sl0_LD, sl1_LD))
            ret_HD.append((sl0_HD, sl1_HD))
            return ret_LD, ret_HD

        if i > 0 and i < Nchks_0 - 1:
            # i in the interior :
            sl0_LD = slice(0, N0 + 2 * b0)  # Slices of LD cube
            sl0_HD = slice(i * N0 - b0, (i + 1) * N0 + b0)  # slices of HD cube
            if j > 0 and j < Nchks_1 - 1:
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
                assert j > 0 and j < Nchks_1 - 1
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
                assert j > 0 and j < Nchks_1 - 1
                sl1_LD = slice(0, N1 + 2 * b1)
                sl1_HD = slice(j * N1 - b1, (j + 1) * N1 + b1)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))

                sl0_LD = slice(N0 + b0, N0 + 2 * b0)
                sl0_HD = slice(0, b0)
                ret_LD.append((sl0_LD, sl1_LD))
                ret_HD.append((sl0_HD, sl1_HD))
                return ret_LD, ret_HD
