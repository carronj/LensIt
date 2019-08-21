from __future__ import print_function

import numpy as np
import sqlite3
import os, io
import pickle as pk
import operator

from lensit import pbs
from lensit.misc.misc_utils import npy_hash

class rng_db:
    """
    Class to save and read random number generators states in a .db file.
    """

    def __init__(self, fname, idtype="INTEGER"):
        if not os.path.exists(fname) and pbs.rank == 0:
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=3600)
            cur = con.cursor()
            cur.execute("create table rngdb (id %s PRIMARY KEY, "
                        "type STRING, pos INTEGER, has_gauss INTEGER,cached_gaussian REAL, keys array)" % idtype)
            con.commit()
        pbs.barrier()

        self.con = sqlite3.connect(fname, timeout=3600., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, id, state):
        try:
            assert (self.get(id) is None)
            self.con.execute("INSERT INTO rngdb (id, type, pos, has_gauss, cached_gaussian, keys) VALUES (?,?,?,?,?,?)",
                             (id, state[0], state[2], state[3], state[4], state[1].reshape(1, len(state[1]))))
            self.con.commit()
        except:
            print("rng_db::rngdb add failed!")

    def get(self, id):
        cur = self.con.cursor()
        cur.execute("SELECT type, pos, has_gauss, cached_gaussian, keys FROM rngdb WHERE id=?", (id,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            assert (len(data) == 5)
            type, pos, has_gauss, cached_gaussian, keys = data
            keys = keys[0]
            return [type, keys, pos, has_gauss, cached_gaussian]

    def delete(self, id):
        try:
            if self.get(id) is None:
                return
            self.con.execute("DELETE FROM rngdb WHERE id=?", (id,))
            self.con.commit()
        except:
            print("rng_db::rngdb delete %s failed!" % id)


class sim_lib(object):
    """
    Generic class for simulations where only rng state is stored.
    np.random rng states are stored in a sqlite3 database.

    By default the rng state function is np.random.get_state.
    The rng_db class is tuned for this state fct, you may need to adapt this.

    Subclass the ._build_sim_from_rng routine and .hashdict and .get_ callables

    jcarron Nov. 2015.
    """

    def __init__(self, lib_dir, get_state_func=np.random.get_state, nsims_max=None):
        if not os.path.exists(lib_dir) and pbs.rank == 0:
            os.makedirs(lib_dir)
        self.nmax = nsims_max
        fn = os.path.join(lib_dir, 'sim_hash.pk')
        if pbs.rank == 0 and not os.path.exists(fn):
            pk.dump(self.hashdict(), open(fn, 'wb'), protocol=2)
        pbs.barrier()

        hash_check(pk.load(open(fn, 'rb')), self.hashdict(), ignore=['lib_dir'])

        self._rng_db = rng_db(lib_dir + '/rngdb.db', idtype='INTEGER')
        self._get_rng_state = get_state_func

    def get_sim(self, idx, **kwargs):
        """ Returns sim number idx """
        if self.has_nmax(): assert idx < self.nmax
        if not self.is_stored(idx):
            # Checks that the sim idx - 1 was previously calculated :
            # if idx > 0 : assert self.is_stored(idx - 1),\
            #    "sim_lib::sim %s absent from the database while calling sim %s"%(str(idx-1),str(idx))
            self._rng_db.add(idx, self._get_rng_state())
        return self._build_sim_from_rng(self._rng_db.get(idx), **kwargs)

    def has_nmax(self):
        return not self.nmax is None

    def is_stored(self, idx):
        """ Checks whether sim idx is stored or not. Boolean output. """
        return not self._rng_db.get(idx) is None

    def is_full(self):
        """ Checks whether all sims are stored or not. Boolean output. """
        if not self.has_nmax(): return False
        for idx in range(self.nmax):
            if not self.is_stored(idx): return False
        return True

    def is_empty(self):
        """ Checks whether any sims is stored. Boolean output. """
        assert self.nmax is not None
        for i in range(self.nmax):
            if self.is_stored(i): return False
        return True

    def hashdict(self):
        """ Subclass this """
        assert 0

    def _build_sim_from_rng(self, rng_state):
        """ Subclass this """
        assert 0


class sim_lib_dat():
    def __init__(self, sim_lib):
        self.sim_lib = sim_lib
        assert hasattr(sim_lib, 'get_dat')

    def get_sim(self, idx): return self.sim_lib.get_dat()

    def get_dat(self): return self.sim_lib.get_dat()

    def hashdict(self):
        return {'sim_lib': self.sim_lib.hashdict(), 'type': 'library_dat'}


class sim_lib_shuffle:
    """
    A sim_lib with remapped indices. sim(i) = input_sim_lib(j(i)).
    E.g. sim_lib_shuffle(sim_lib,shuffle = lambda idx : 0)
    """

    def __init__(self, sim_lib, shuffle=lambda idx: idx):
        self.sim_lib = sim_lib
        self._shuffle = shuffle

    def get_dat(self):
        assert hasattr(self.sim_lib, 'get_dat')
        return self.sim_lib.get_dat()

    def get_sim(self, idx): return self.sim_lib.get_sim(self._shuffle(idx))

    def get_shuffle_func(self): return self._shuffle

    def hashdict(self):
        #FIXME:
        hash = {}
        for i in range(100): hash[i] = self._shuffle(i)
        return {'sim_lib': self.sim_lib.hashdict(), 'shuffle': hash}


class sim_lib_sum():
    """
    return sums of sim_libs, with weights if provided.
    Change operator.iadd to operator.something_else to change the operation, e.g. a multiplication.

    All inputs must be sim_lib instances with a get_sim and get_hashdict callables.
    By default returns sum_i sim_i *weight_i

    Does not store anything, no need for lib_dir and suchs.
    """

    def __init__(self, sim_lib_list, operation=operator.iadd, weights=None):
        if weights is not None: assert (len(sim_lib_list) == len(weights)), "sim_lib_sum::inputs not understood"
        self.sim_lib_list = sim_lib_list
        self.weights = weights
        self.operator = operation

    def get_sim(self, idx):
        """ Returns sim number idx by applying the self.operator to the maps """
        if self.has_weights():
            sim = self.sim_lib_list[0].get_sim(idx) * self.weights[0]
            for w, sim_lib in zip(self.weights[1:], self.sim_lib_list[1:]):
                sim = self.operator(sim, sim_lib.get_sim(idx) * w)
        else:
            sim = self.sim_lib_list[0].get_sim(idx)
            for sim_lib in self.sim_lib_list[1:]:
                sim = self.operator(sim, sim_lib.get_sim(idx))
        return sim

    def has_weights(self):
        return self.weights is not None

    def hashdict(self):
        hash = {'weights': self.weights}
        for i, sim_lib in enumerate(self.sim_lib_list):
            hash['sim_lib_%s' % i] = sim_lib.hashdict()
        return hash


class Gauss_sim_generic(sim_lib):
    """
    Class for simple Gaussian sims in Euclidean space of any dimension.
    The input pk (cl) array is simply interpolated on the frequency grid, make sure to include
    all ingredients you want. You can add a keyword to include a pixwin fct.
    Derives from sim_lib and links to a sqlite3 db containing the rng states of each sim.

    Ex :
    res = 9
    shape = (2**res,2**res)
    lside = 10.*np.pi/180.*np.ones(2)

    lib_cmb_unl = sims.Gauss_sim_generic(lib_dir,cl_unl,shape,lside)
    sim = lib_cmb_unl.get_sim(0)
    """

    def __init__(self, lib_dir, cl, shape, lsides, with_pixwin=False, **kwargs):
        self.lib_dir = lib_dir
        assert (cl >= 0.).all()
        self.cl = cl
        self.shape = shape
        self._with_pixwin = with_pixwin
        assert (len(shape) == len(lsides))
        self.lsides = tuple(lsides)
        super(Gauss_sim_generic, self).__init__(lib_dir, **kwargs)
        if self.is_empty() and len(cl) - 1 <= self.kmax_scal():
            print("Gauss_sim_generic::Warning, grid kmax larger than kmax provided. These will be set to zero.")
            print(round(self.kmax_scal()), len(cl) - 1)

    def ndim(self):
        return len(self.shape)

    def kmin(self):
        return (2 * np.pi) / np.array(self.lsides)

    def kmax(self):
        return np.pi * np.array(self.shape) / np.array(self.lsides)

    def kmax_scal(self):
        return np.sqrt(np.sum(self.kmax() ** 2))

    def has_pixwin(self):
        return self._with_pixwin

    def _rfftreals(self):
        # TODO : for any dim.
        assert len(self.shape) == 2
        N0, N1 = self.shape
        fx = [0];
        fy = [0]
        if N0 % 2 == 0: fx.append(N0 / 2); fy.append(0)
        if N1 % 2 == 0: fx.append(0); fy.append(N1 / 2)
        if N1 % 2 == 0 and N0 % 2 == 0: fx.append(N0 / 2); fy.append(N1 / 2)
        return np.array(fx), np.array(fy)

    def _Freq(self, i, N):
        """
        Outputs the absolute integers frequencies [0,1,...,N/2,N/2-1,...,1]
        in numpy fft convention as integer i runs from 0 to N-1.
        Inputs can be numpy arrays e.g. i (i1,i2,i3) with N (N1,N2,N3)
                                  or i (i1,i2,...) with N
        Both inputs must be integers.
        All entries of N must be even.
        """
        assert (np.all(N % 2 == 0)), "This routine only for even numbers of points"
        return i - 2 * (i >= (N / 2)) * (i % (N / 2))

    def _outerproducts(self, vs):
        """
        vs is a list of 1d numpy arrays, not necessarily of the same size.
        Return a matrix A_i1_i2..i_ndim = vi1_vi2_..v_indim.
        Use np.outer recursively on flattened arrays.
         """

        # check input and infer new shape  :
        assert (isinstance(vs, list)), "Want list of 1d arrays"
        ndim = len(vs)
        if ndim == 1: return vs[0]
        shape = ()
        for i in range(ndim):
            assert (vs[i].ndim == 1), "Want list of 1d arrays"
            shape += (vs[i].size,)

        B = vs[ndim - 1]

        for i in range(1, ndim): B = np.outer(vs[ndim - 1 - i], B).flatten()
        return B.reshape(shape)

    def _square_pixwin_map(self, shape):
        """
        pixel window function of square top hat for any dimenson
        """
        vs = []
        for ax in range(len(shape)):
            lcell_ka = 0.5 * self._Freq(np.arange(shape[ax]), shape[ax]) * (2. * np.pi / shape[ax])
            vs.append(np.insert(np.sin(lcell_ka[1:]) / lcell_ka[1:], 0, 1.))
        return self._outerproducts(vs)

    def _sqd_freqmap(self):
        """
        Returns the array of squared frequencies, in grid units.
        """
        kmin2 = self.kmin() ** 2
        s = self.shape
        # First we check if the cube is regular
        if len(np.unique(s)) == 1 and len(np.unique(self.lsides)) == 1:
            # regular hypercube simplifies matters.
            l02 = self._Freq(np.arange(s[0]), s[0]) ** 2 * kmin2[0]
            ones = np.ones(s[0])
            if self.ndim == 1: return l02
            vec = [l02]
            for i in range(1, self.ndim()):
                vec.append(ones)
            l0x2 = self._outerproducts(vec)
            sqd_freq = np.zeros(s)
            for i in range(0, self.ndim()):
                sqd_freq += np.swapaxes(l0x2, 0, i)
            return sqd_freq
        # Ok, that's fine, let's use a different dumb method.
        idc = np.indices(s)
        mapk = self._Freq(idc[0, :], s[0]) ** 2 * kmin2[0]
        for j in range(1, self.ndim()):
            mapk += self._Freq(idc[j, :], s[j]) ** 2 * kmin2[j]
        return mapk

    def hashdict(self):
        return {'lib_dir': self.lib_dir, 'shape': self.shape, 'lsides': self.lsides,
                'has_pixwin': self.has_pixwin(), 'cl': npy_hash(self.cl)}

    def _build_sim_from_rng_2(self, rng_state):
        """
        Returns one realisations of a G. field with the input cl.
        The code actually calculates two and throws away one for pure laziness. But these things look so cheap for 2d.
        As a technicality, the keyword right = 0 is important, in order not to add unwanted white noise to the sim.
        """
        N_rootV = np.prod(self.shape / np.sqrt(self.lsides))
        root_spec_map = np.interp(self._sqd_freqmap().flatten(), np.arange(len(self.cl)) ** 2,
                                  np.sqrt(self.cl) * N_rootV, right=0., left=0.)
        if self.has_pixwin(): root_spec_map *= np.sqrt(self._square_pixwin_map(self.shape)).flatten()
        np.random.set_state(rng_state)
        sims = (1j * np.random.normal(size=np.prod(self.shape)) + np.random.normal(size=np.prod(self.shape))) \
               * root_spec_map
        return np.fft.ifft2(sims.reshape(self.shape)).real

    def _build_sim_from_rng(self, rng_state):
        """
        Returns one realisation of a G. field with the input cl.
        As a technicality, the keyword right = 0 is important, in order not to add unwanted white noise to the sim.
        """
        N_rootV = np.prod(self.shape / np.sqrt(self.lsides))
        last_axis = len(self.shape) - 1
        sims = np.take(self._sqd_freqmap(), np.arange(self.shape[-1] / 2 + 1), axis=last_axis)
        sims = np.interp(sims, np.arange(len(self.cl)) ** 2, np.sqrt(self.cl / 2) * N_rootV, right=0., left=0.)
        if self.has_pixwin():
            sims *= np.sqrt(
                np.take(self._square_pixwin_map(self.shape), np.arange(self.shape[-1] / 2 + 1), axis=last_axis))
        np.random.set_state(rng_state)
        rfft_shape = np.array(self.shape)
        rfft_shape[-1] = self.shape[-1] / 2 + 1
        sims = (1j * np.random.normal(size=rfft_shape) + np.random.normal(size=rfft_shape)) * sims
        # Corrects for pure real frequencies and redundant frequencies :
        sla = slice(self.shape[0] / 2 + 1, self.shape[0], 1)
        slb = slice(self.shape[0] / 2 - 1, 0, -1)

        sims.real[sla, [-1, 0]] = sims.real[slb, [-1, 0]]
        sims.imag[sla, [-1, 0]] = -sims.imag[slb, [-1, 0]]

        sims.imag[self._rfftreals()] = 0.
        sims.real[self._rfftreals()] *= np.sqrt(2.)
        return np.fft.irfft2(sims, self.shape)


class apply_sim_lib:
    """
    Generic class returning sim_b.apply(sim_a).
    Sims of the apply_sim_lib must a have .apply(map) callable.

    For instance, the following lines :

    lib_cmb_unl = sims.Gauss_sim_generic('./testtemp/cl_unl',cl_unl,shape,lside,with_pixwin=True)
    lib_displ = sims.Gauss_displ_2dsim('./testtemp/displ',cl_pp,shape,lside)
    lib_cmb_len = sims.apply_sim_lib(lib_cmb_unl,lib_displ)

    gives a library for lensed CMB's
    """

    def __init__(self, base_sim_lib, apply_sim_lib):
        self.base_sims = base_sim_lib
        self.apply_sims = apply_sim_lib

    def hashdict(self):
        return {'base_sim_lib': self.base_sims.hashdict(), 'apply_sim_lib': self.apply_sims.hashdict()}

    def get_sim(self, idx):
        disp = self.apply_sims.get_sim(idx)
        return disp.apply(self.base_sims.get_sim(idx))


def hash_check(hash1, hash2, ignore=None, keychain=None):
    """ from Mr. DH """
    if ignore is None: ignore = []
    if keychain is None: keychain = []
    keys1 = hash1.keys()
    keys2 = hash2.keys()

    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)

    for key in set(keys1).union(set(keys2)):
        v1 = hash1[key]
        v2 = hash2[key]

        def hashfail(msg=None):
            print("ERROR: HASHCHECK FAIL AT KEY = " + ':'.join(keychain + [key]))
            if msg is not None:
                print("   ", msg)
            print("   ", "V1 = ", v1)
            print("   ", "V2 = ", v2)
            assert 0

        if type(v1) != type(v2):
            hashfail('UNEQUAL TYPES')
        elif type(v2) == dict:
            hash_check(v1, v2, ignore=ignore, keychain=keychain + [key])
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not (v1 == v2):
                hashfail('UNEQUAL VALUES')


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0);
    return buffer(out.read())


sqlite3.register_adapter(np.ndarray, adapt_array)


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_converter("array", convert_array)
