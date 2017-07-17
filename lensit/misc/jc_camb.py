__author__ = 'jcarron'
# jcarron, Brighton, May 2015
"""
Some utils to call CAMB and/or read CAMB outputs.
"""

import cls as cosmo
import numpy as np
import os
import readcol as rc

PathToCamb = '/Users/jcarron/camb'
PathToCambParamFile = '/Users/jcarron/PyCharmProjects/jc_camb'
typedict = ['matterpower', 'scalCls', 'lenspotentialCls', 'lensedCls', 'tensCls']


# def getPk_fromparams(**kwargs):
#    run_camb_fromparams(**kwargs)
#    return spectra_fromcambfile(PathToCambParamFile + 'camb_output_matterpower_1.dat' )

def get_matterpks_fromcamb(zs_, camb_params):
    # Returns a list of matter pks instances for the given redshifts, in increasing order of redshift.
    zs = np.sort(zs_)[::-1]
    files = []
    # Adapt camb parameter file
    camb_params['transfer_num_redshifts'] = len(zs)
    camb_params['get_transfer'] = 'T'
    camb_params['get_scalar_cls'] = 'F'

    for i, z in enumerate(zs):
        files.append(camb_params['output_root'] + '_matterpower_' + str(i + 1) + '.dat')
        key = 'transfer_redshift(' + str(i + 1) + ')'
        camb_params[key] = z
    run_camb_fromparams(camb_params)
    Pks = []
    for file in files:
        assert (os.path.exists(file)), "Output file not found, something went wrong."
        Pks.insert(0, spectra_fromcambfile(file, type='matterpower'))
    return Pks


def spectra_fromcambfile(file, type=None, lmax=None):
    """
    Returns a bunch of spectra from a CAMB output file
    'type' is either a tranfer Pk file, a unlensed Cls or lensed Cls file
    following CAMB conv. for output files
    """
    if type is None:  # Let's try to guess from the file name
        for s in typedict:
            if s in file: type = s
    assert (type in typedict), type
    # reads a list of columns. Should find the data type by itself.
    cols = rc.readcol(file, twod=False)
    if type == 'matterpower':  # Outputs a jc.cosmo Pk instance.
        assert (len(cols) >= 2), len(cols)
        return cosmo.Pk(cols[0], cols[1])
    elif type == 'lenspotentialCls':  # seven jc.cosmo Cl instances
        assert (len(cols) >= 8), len(cols)
        ell = cols[0]
        w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
        idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell))
        Cltt = cosmo.Cl_lminlmax(ell[idc], cols[1][idc] / w[idc])
        Clee = cosmo.Cl_lminlmax(ell[idc], cols[2][idc] / w[idc])
        Clbb = cosmo.Cl_lminlmax(ell[idc], cols[3][idc] / w[idc])
        Clte = cosmo.Cl_lminlmax(ell[idc], cols[4][idc] / w[idc])
        w = ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        Clpp = cosmo.Cl_lminlmax(ell[idc], cols[5][idc] / w[idc])
        w = np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        Clpt = cosmo.Cl_lminlmax(ell[idc], cols[6][idc] / w[idc])
        Clpe = cosmo.Cl_lminlmax(ell[idc], cols[7][idc] / w[idc])
        return {'tt': Cltt, 'ee': Clee, 'te': Clte, 'bb': Clbb, 'pp': Clpp, 'pt': Clpt, 'pe': Clpe}
    elif type == 'lensedCls' or type == 'tensCls':  # 4 jc_cosmo Cl instances
        assert (len(cols) >= 5), len(cols)
        ell = cols[0]
        w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
        idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell))
        Cltt = cosmo.Cl_lminlmax(ell[idc], cols[1][idc] / w[idc])
        Clee = cosmo.Cl_lminlmax(ell[idc], cols[2][idc] / w[idc])
        Clbb = cosmo.Cl_lminlmax(ell[idc], cols[3][idc] / w[idc])
        Clte = cosmo.Cl_lminlmax(ell[idc], cols[4][idc] / w[idc])
        return {'tt': Cltt, 'ee': Clee, 'bb': Clbb, 'te': Clte}

    assert (0), "How did you get there ??"
    return 0


def read_params(paramfile):
    assert os.path.exists(paramfile), paramfile
    params = {}
    with open(paramfile) as f:
        for line in f:
            (key, temp, val) = line.split()
            params[key] = val
    return params


def set_cambparams(**kwargs):
    """
    Creates a dictionary with CAMB parameter to input the code.
    First sets up a fiducial model and check for replacement from **kwargs
    """
    params = \
        {
            'output_root': PathToCambParamFile + '/camb_output',
            'get_scalar_cls': 'T',
            'get_vector_cls': 'F',
            'get_tensor_cls': 'F',
            'CMB_outputscale': 7.42835025e+12,
            'get_transfer': 'T',
            'transfer_kmax': 10.,
            'transfer_k_per_logint': 0,
            'transfer_high_precision': 'T',
            'transfer_num_redshifts': 1,
            'transfer_interp_matterpower': 'T',
            'transfer_filename(1)': 'transfer_out.dat',
            'transfer_redshift(1)': 0,
            'accuracy_boost': 1,
            'l_accuracy_boost': 1,
            'high_accuracy_default': 'T',
            'do_nonlinear': 3,
            'l_max_scalar': 5500,
            'k_eta_max_scalar': 100000,
            'do_lensing': 'T',
            'lensing_method': 1,
            'w': -1.0,
            'cs2_lam': 1,
            'hubble': 67.86682,
            'use_physical': 'T',
            'ombh2': 0.02227716,
            'omch2': 0.1184293,
            'omnuh2': 0.0006451439,
            'omk': 0,
            'temp_cmb': 2.7255,
            'helium_fraction': 0.245352,
            'massless_neutrinos': 2.03066666667,
            'nu_mass_eigenstates': 1,
            'massive_neutrinos': 1,
            'share_delta_neff': 'F',
            'nu_mass_degeneracies': 1.01533333333,
            'nu_mass_fractions': 1,
            'halofit_version': 4,
            'DebugParam': 0.000000000000000E+000,
            'Alens': 1.00000000000000,
            'reionization': 'T',
            're_use_optical_depth': 'T',
            're_optical_depth': 0.06664549,
            're_delta_redshift': 0.5,
            're_ionization_frac': -1,
            'pivot_scalar': 0.05,
            'pivot_tensor': 0.05,
            'initial_power_num': 1,
            'scalar_spectral_index(1)': 0.9682903,
            'scalar_nrun(1)': 0.0,
            'scalar_nrunrun(1)': 0,
            'scalar_amp(1)': 2.140509e-09,
            'RECFAST_fudge_He': 0.86,
            'RECFAST_Heswitch': 6,
            'RECFAST_Hswitch': 'T',
            'RECFAST_fudge': 1.14,
            'AGauss1': -0.140000000000000,
            'AGauss2': 7.900000000000000E-002,
            'zGauss1': 7.28000000000000,
            'zGauss2': 6.73000000000000,
            'wGauss1': 0.180000000000000,
            'wGauss2': 0.330000000000000,
            'do_lensing_bispectrum': 'F',
            'do_primordial_bispectrum': 'F',
            'initial_condition': 1,
            'scalar_output_file': 'scalCls.dat',
            'lensed_output_file': 'lensedCls.dat',
            'lens_potential_output_file': 'lenspotentialCls.dat',
            'accurate_polarization': 'T',
            'accurate_reionization': 'T',
            'accurate_BB': 'F',
            'derived_parameters': 'T',
            'version_check': 'Jan15',
            'do_late_rad_truncation': 'T',
            'feedback_level': 1,
            'massive_nu_approx': 1,
            'number_of_threads': 0,
            'use_spline_template': 'T',
            'l_sample_boost': 1, }
    # replacing keywords :
    for key, value in kwargs.iteritems():
        if params.has_key(key):
            params[key] = value
        else:
            print "Key not found : adding to param. file :", key, str(value)
            params[key] = value
    return params


def get_lensedcls(params):
    run_camb_fromparams(params)
    return spectra_fromcambfile(params['output_root'] + '_' + params['lensed_output_file'])


def get_partiallylensedcls(params, w):
    # Produces spectra lensed with w_L * cpp_L
    params['lensing_method'] = 4
    ell = np.arange(len(w), dtype=int)
    np.savetxt('/Users/jcarron/camb/cpp_weights.txt', np.array([ell, w]).transpose(), fmt=['%i', '%10.5f'])
    run_camb_fromparams(params)
    return spectra_fromcambfile(params['output_root'] + '_' + params['lensed_output_file'])


def run_camb_fromparams(params):
    """
    run camb from keywords adapting the fiducial params.ini
    """
    write_camb_parameterfile(params)
    run_camb_fromparameterfile(PathToCambParamFile + '/params.ini')


def write_camb_parameterfile(params):
    """
    write the camb params dictionary in the file PathToCambParamFile/params.ini
    :param params: camb parameter dictionary
    :return:
    """
    path = PathToCambParamFile
    if not os.path.exists(path): os.mkdir(path)
    f = open(path + '/params.ini', 'w')
    align = 30
    for key, value in sorted(params.items()):
        line = key + (' ' * (np.max((align - len(key), 0)))) + ' = ' + str(value)
        f.write(line)
        f.write("\n")
    f.close()


def run_camb_fromparameterfile(parameterfile):
    curdir = os.getcwd()
    os.chdir(PathToCamb)  # This seems to be necessary for a correct CAMB run ?
    from subprocess import call
    call(['./camb', parameterfile])
    os.chdir(curdir)
