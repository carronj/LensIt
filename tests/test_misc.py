# pytest --ignore=./scripts --ignore=./lensit
import os
import numpy as np
assert 'LENSIT' in os.environ.keys()

def test_lencmbs():
    import lensit as li
    from lensit.ffs_deflect import ffs_deflect
    lib = li.get_lencmbs_lib(res=8, cache_sims=False, nsims=120)
    plm = lib.get_sim_plm(0)
    f = ffs_deflect.displacement_fromplm(lib.lib_skyalm, plm)
    assert 1

def test_inverse():
    import lensit as li
    from lensit.ffs_deflect import ffs_deflect
    lib = li.get_lencmbs_lib(res=8, cache_sims=False, nsims=120)
    plm = lib.get_sim_plm(0)
    f = ffs_deflect.displacement_fromplm(lib.lib_skyalm, plm, verbose=False)
    fi = f.get_inverse()
    assert 1

def test_maps():
    import lensit as li
    lib = li.get_maps_lib('S4', 8, 8)
    lib.get_sim_tmap(0)
    assert 1

def test_cl():
    import lensit as li
    cl_unl, cl_len = li.get_fidcls(ellmax_sky=5000)
    assert len(cl_unl['tt'] == 5001)
    assert 1

def test_biases():
    import lensit as li
    isocov = li.get_isocov('S4', 8, 8)
    N0s = isocov.get_N0cls('QU', isocov.lib_skyalm)
    assert np.all(N0s[0][:3000] >= 0.)
    assert np.all(N0s[1][:3000] >= 0.)
    ell, = np.where(isocov.lib_skyalm.get_Nell()[:3001] == 1)
    assert np.all(N0s[0][ell] == 0.)
    assert np.all(N0s[1][ell] == 0.)


def test_iters4():
    import lensit as li
    from lensit.ffs_iterators.ffs_iterator import ffs_iterator_pertMF
    from lensit.misc.misc_utils import gauss_beam
    from lensit.qcinv import ffs_ninv_filt_ideal, chain_samples
    from lensit.ffs_covs import ell_mat
    def get_starting_point(idx):
        sims = li.get_maps_lib('S4', 10, 11, nsims=1)  # Simulation-library for configuration 'S4'.
        # Parameters 10, 11 produces data on 645 sq. deg,
        # with lensed CMB's generated at 0.75 arcmin resolution,
        # but data collected at 1.5 arcmin resolution.

        isocov = li.get_isocov('S4', 10, 11)  # Isotropic filtering instance, that can used for Q.E. calculation
        # and other things.isocov.lib_datalm defines the mode-filtering applied
        # the data, and isocov.lib_skyalm the band-limits of the unlensed sky.
        print(" I will be using data from ell=%s to ell=%s only" % (isocov.lib_datalm.ellmin, isocov.lib_datalm.ellmax))
        print(" The sky band-limit is ell=%s" % (isocov.lib_skyalm.ellmax))

        lib_qlm = isocov.lib_skyalm  #: This means we will reconstruct the lensing potential for all unlensed sky modes.

        def cli(cl):
            ret = np.zeros_like(cl)
            ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
            return ret

        # We now build the Wiener-filtered quadratic estimator. We use lensed CMB spectra in the weights.
        datalms = np.array([isocov.lib_datalm.map2alm(m) for m in sims.get_sim_qumap(idx)])
        H0len = cli(isocov.get_N0cls('QU', lib_qlm, use_cls_len=True)[0])
        plm0 = 0.5 * isocov.get_qlms('QU', isocov.get_iblms('QU', datalms, use_cls_len=True)[0], lib_qlm,
                                     use_cls_len=True)[0]

        # Normalization and Wiener-filtering:
        cpp_prior = li.get_fidcls()[0]['pp'][:lib_qlm.ellmax + 1]
        lib_qlm.almxfl(plm0, cli(H0len + cli(cpp_prior)), inplace=True)

        # Initial likelihood curvature guess. We use here N0 as calculated with unlensed CMB spectra:
        H0unl = cli(isocov.get_N0cls('QU', lib_qlm, use_cls_len=False)[0])
        return plm0, lib_qlm, datalms, isocov.lib_datalm, H0unl, H0len


    def get_itlib(lib_dir, plm0, lib_qlm, datalms, lib_datalm, H0, beam_fwhmamin=3., NlevT_filt=1.5,
                  NlevP_filt=1.5 * np.sqrt(2.)):
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        # Prior on lensing power spectrum, and CMB spectra for the filtering at each iteration step.
        cls_unl = li.get_fidcls(6000)[0]
        cpp_prior = cls_unl['pp'][:]

        lib_skyalm = li.ffs_covs.ell_mat.ffs_alm_pyFFTW(lib_datalm.ell_mat, filt_func=lambda ell: ell <= 6000)
        #: This means we perform here the lensing of CMB skies at the same resolution
        #  than the data with the band-limit of 6000.
        transf = gauss_beam(beam_fwhmamin / 180. / 60. * np.pi, lmax=6000)  #: fiducial beam

        # Anisotropic filtering instance, with unlensed CMB spectra as inputs. Delfections will be added by the iterator.
        filt = li.qcinv.ffs_ninv_filt_ideal.ffs_ninv_filt(lib_datalm, lib_skyalm, cls_unl, transf, NlevT_filt, NlevP_filt)

        # Description of the multi-grid chain to use: (here the simplest, diagonal pre-conditioner)
        chain_descr = li.qcinv.chain_samples.get_isomgchain(filt.lib_skyalm.ellmax, filt.lib_datalm.shape,
                                                            tol=1e-6, iter_max=200)

        # We assume no primordial B-modes, the E-B filtering will assume all B-modes are either noise or lensing:
        opfilt = li.qcinv.opfilt_cinv_noBB
        opfilt._type = 'QU'  # We consider polarization only

        # With all this now in place, we can build the iterator instance:
        iterator = ffs_iterator_pertMF(lib_dir, 'QU', filt, datalms, lib_qlm,
                                       plm0, H0, cpp_prior, chain_descr=chain_descr, opfilt=opfilt, verbose=True)
        # We use here an iterator instance that uses an analytical approximation
        # for the mean-field term at each step.
        return iterator



    plm0, lib_qlm, datalms, lib_datalm, H0, H0len = get_starting_point(0)

    lib_dir = os.path.join(os.environ['LENSIT'], 'temp', '_testiterator_S4_sim%03d_456EF' % 0)
    assert not os.path.exists(lib_dir), lib_dir
    itlib = get_itlib(lib_dir, plm0, lib_qlm, datalms, lib_datalm, H0)
    itlib.soltn_cond = True
    for i in range(11):
        itlib.iterate(i, 'p')

    from subprocess import call
    call(['cat', os.path.join(itlib.lib_dir, 'history_increment.txt')])
    gradno  =np.loadtxt(os.path.join(itlib.lib_dir, 'history_increment.txt')).transpose()[3]
    assert gradno[-1] < 0.01
    assert np.all(np.sort(gradno) == gradno[::-1])
    import shutil
    shutil.rmtree(itlib.lib_dir)

if __name__ == '__main__':
    test_lencmbs()
    test_inverse()
    test_maps()
    test_cl()
    test_iters4()