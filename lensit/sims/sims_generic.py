from __future__ import print_function

import numpy as np
import sqlite3
import six
import os, io
import pickle as pk
import operator

from lensit.pbs import pbs

def adapt_array(arr):
    out = io.BytesIO(); np.save(out, arr)
    out.seek(0)
    return buffer(out.read()) if six.PY2 else memoryview(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

class rng_db:
    """ Class to save and read random number generators states in a sqlite database file.

    """

    def __init__(self, fname, idtype="INTEGER"):
        if not os.path.exists(fname) and pbs.rank == 0:
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=3600)
            cur = con.cursor()
            cur.execute("create table rngdb (id %s PRIMARY KEY, "
                        "type STRING, pos INTEGER, has_gauss INTEGER,cached_gaussian REAL, keys STRING)" % idtype)
            con.commit()
        pbs.barrier()

        self.con = sqlite3.connect(fname, timeout=3600., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, idx, state):
        try:
            assert (self.get(idx) is None)
            keys_string = '_'.join(str(s) for s in state[1])
            self.con.execute("INSERT INTO rngdb (id, type, pos, has_gauss, cached_gaussian, keys) VALUES (?,?,?,?,?,?)",
                             (idx, state[0], state[2], state[3], state[4], keys_string))
            self.con.commit()
        except:
            print("rng_db::rngdb add failed!")

    def get(self, idx):
        cur = self.con.cursor()
        cur.execute("SELECT type, pos, has_gauss, cached_gaussian, keys FROM rngdb WHERE id=?", (idx,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            assert (len(data) == 5)
            typ, pos, has_gauss, cached_gaussian, keys = data
            keys = np.array([int(a) for a in keys.split('_')], dtype=np.uint32)
            return [typ, keys, pos, has_gauss, cached_gaussian]

    def delete(self, idx):
        try:
            if self.get(idx) is None:
                return
            self.con.execute("DELETE FROM rngdb WHERE id=?", (idx,))
            self.con.commit()
        except:
            print("rng_db::rngdb delete %s failed!" % idx)

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

        self._rng_db = rng_db(os.path.join(lib_dir, 'rngdb.db'), idtype='INTEGER')
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


class sim_lib_dat:
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


class sim_lib_sum:
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
        h = {'weights': self.weights}
        for i, sim_lib in enumerate(self.sim_lib_list):
            h['sim_lib_%s' % i] = sim_lib.hashdict()
        return h


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