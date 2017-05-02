import cd_solve
import numpy as np


def get_defaultmgchain(lmax_sky, lsides, datshape, tol=1e-5, iter_max=np.inf, dense_file='', **kwargs):
    # FIXME :
    assert datshape[0] == datshape[1], datshape
    nside_max = datshape[0]
    if lmax_sky > 4000:
        dense_size = 2000
        if np.prod(lsides) >= (4. * np.pi) - 0.1:
            lmax_dense = 64
        else:
            lmax_dense = np.sqrt(2. / 2. / np.pi * (2 * np.pi) ** 2 / np.prod(lsides) * dense_size)
            lmax_dense = int(np.round(min(lmax_dense, 1300)))
        print "chain_samples : setting lmax_dense to ", lmax_dense
        chain_descr = [
            [3, ["split(dense(" + dense_file + "), %s, diag_cl)" % (int(lmax_dense))], 1400, nside_max / 4, 3, 0.,
             cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [2, ["split(stage(3), %s, diag_cl)" % (1400)], 3000, nside_max / 2, 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [1, ["split(stage(2), %s, diag_cl)" % (3000)], 4000, nside_max / 2, 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [0, ["split(stage(1), %s, diag_cl)" % (4000)], lmax_sky, nside_max, iter_max, tol, cd_solve.tr_cg,
             cd_solve.cache_mem()]]
    elif lmax_sky > 3000:
        dense_size = 2000
        lmax_dense = np.sqrt(2. / 2. / np.pi * (2 * np.pi) ** 2 / np.prod(lsides) * dense_size)
        lmax_dense = int(np.round(min(lmax_dense, 1300)))
        print "chain_samples : setting lmax_dense to ", lmax_dense
        chain_descr = [
            [2, ["split(dense(" + dense_file + "), %s, diag_cl)" % (int(lmax_dense))], 1400, nside_max / 4, 3, 0.,
             cd_solve.tr_cg, cd_solve.cache_mem()],
            [1, ["split(stage(2), %s, diag_cl)" % (1400)], 3000, nside_max / 2, 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [0, ["split(stage(1), %s, diag_cl)" % (3000)], lmax_sky, nside_max / 2, iter_max, tol, cd_solve.tr_cg,
             cd_solve.cache_mem()]]
    else:
        # Same as PL2015 pipeline :
        # chain_descr = [[3, ["split(dense(" + pcf + "), 64, diag_cl)"], 256, 128, 3, 0.0, qcinv.cd_solve.tr_cg,
        #                qcinv.cd_solve.cache_mem()],
        #               [2, ["split(stage(3),  256, diag_cl)"], 512, 256, 3, 0.0, qcinv.cd_solve.tr_cg,
        #                qcinv.cd_solve.cache_mem()],
        #               [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, qcinv.cd_solve.tr_cg,
        #                qcinv.cd_solve.cache_mem()],
        #               [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, qcinv.cd_solve.tr_cg,
        #                qcinv.cd_solve.cache_mem()]]
        # On the full flat sky with lmax_sky 2048 it solves the thing to 1e-5 in 8 min or so on the laptop.
        res = lambda fac: max(10, nside_max / fac)
        chain_descr = [
            [3, ["split(dense(" + dense_file + "), %s, diag_cl)" % (64)], 256, res(16), 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [2, ["split(stage(3), %s, diag_cl)" % (256)], 512, res(8), 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [1, ["split(stage(2), %s, diag_cl)" % (512)], 1024, res(4), 3, 0., cd_solve.tr_cg,
             cd_solve.cache_mem()],
            [0, ["split(stage(1), %s, diag_cl)" % (1024)], lmax_sky, nside_max, iter_max, tol, cd_solve.tr_cg,
             cd_solve.cache_mem()]]

    return chain_descr


def get_isomgchain(lmax_sky, datshape, tol=1e-5, iter_max=np.inf, **kwargs):
    assert datshape[0] == datshape[1], datshape
    nside_max = datshape[0]
    return [[0, ["diag_cl"], lmax_sky, nside_max, iter_max, tol, cd_solve.tr_cg, cd_solve.cache_mem()]]
