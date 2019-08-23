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


if __name__ == '__main__':
    test_lencmbs()
    test_inverse()
    test_maps()
    test_cl()