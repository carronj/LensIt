# pytest --ignore=./scripts --ignore=./lensit
import os
assert 'LENSIT' in os.environ.keys()

def test_lencmbs():
    import lensit as li
    lib = li.get_lencmbs_lib(res=8, cache_sims=False, nsims=120)
    plm = lib.get_sim_plm(0)
    assert 1

def test_maps():
    import lensit as li
    lib = li.get_maps_lib('S4', 8, 8)
    lib.get_sim_tmap(0)
    assert 1

def test_cl():
    import lensit as li
    cl_unl, cl_len = li.get_fidcls()
    assert 1
