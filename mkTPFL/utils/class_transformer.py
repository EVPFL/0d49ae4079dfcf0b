"""A module to keep track of parameters for the CKKS scheme."""
from HEAAN import ZZ, Uint64_t, ComplexDouble, Plaintext, SecretKey, Key, Ciphertext, Ciphertexts
from mkTPFL.ckks.ckks_encoder import CKKSMessage, CKKSEncoder

from MKLHS import LHSSKey, LHSPKey
from mkTPFL.ckks.ckks_parameters import *

from LHH import LHHVal, LHHHash
from mkTPFL.utils.lhh_parameters import LHH_init

import numpy as np
import time
import pickle
from math import ceil
from collections import defaultdict
from fedAvg.utils.worker_utils import mkdir


#### ------------------------------------------------------------
#### --------------- WRITE & READ DATA FROM FILE  ---------------
#### ------------------------------------------------------------ 
def saveData(data, data_name=None, file_path=None, print_flag=False):
    begin_time = time.time()
    if not file_path:
        dir_path = mkdir('./result/datas/')
        file_path = str(dir_path)+str(data_name)+'_data.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    if print_flag:
        print('  >>> write ['+str(file_path)+'], using '
                +str(round(time.time()-begin_time, 4))+' seconds')

def getData(data_name=None, file_path=None, print_flag=False):
    begin_time = time.time()
    if not file_path:
        dir_path = mkdir('./result/datas/')
        file_path = str(dir_path)+str(data_name)+'_data.pkl'
    with open(file_path, 'rb') as inf:
        data = pickle.load(inf)
    if print_flag:
        print('  >>> read ['+str(file_path)+'], using '
                +str(round(time.time()-begin_time, 4))+' seconds')
    return data

def getObjFromFile(file_path, class_type, print_flag=False):
    data = getData(file_path=file_path, print_flag=print_flag)
    res_obj = data2Obj(data, class_type)
    return res_obj
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 

#### ------------------------------------------------------------
#### ----------------  OBJECT & DATA TRANSFORMER ----------------
#### ------------------------------------------------------------ 
#### ----------------- TRANSFER OBJECT TO DATA ------------------
class DataCiphertexts():
    """ An instance of Ciphertexts for saving. """
    def __init__(self, ciphs, print_flag=False):
        """ Inits DataCiphertexts with the given Ciphertexts/Ciphertext(s). """
        begin_time = time.time()
        if print_flag:
            print('  >>> transfer Ciphertexts to DataCiphertexts ...')

        if isinstance(ciphs, Ciphertexts):
            self.size = ciphs.size
            self.logp = ciphs.logp
            self.logq = ciphs.logq
            self.n = ciphs.n
            self.ax = DataZZ(ciphs.ax)
            self.bxs = DataZZ(ciphs.bxs, size=ciphs.size*N)
            shape = self.size
        else:
            if isinstance(ciphs, Ciphertext):
                ciphs = [ciphs]
            ciphs = np.array(ciphs)
            shape = ciphs.shape
            assert len(shape) == 1 or len(shape) == 2
            c_example = ciphs[0] if isinstance(ciphs[0], Ciphertext) else ciphs[0][0]
            assert isinstance(c_example, Ciphertext)
            self.logp = c_example.logp
            self.logq = c_example.logq
            self.n = c_example.n
            self.shape = shape
            self.axs = np.zeros(shape=shape)
            self.bxs = np.zeros(shape=shape)

            fixed_ax_flag = False
            if len(shape) == 1:
                axs = [c.ax for c in ciphs]
                bxs = [c.bx for c in ciphs]
                set_axs = set(axs)
                fixed_ax_flag = True if len(set_axs)==1 else False
                self.axs = [DataZZ(axs[0])] if fixed_ax_flag else [DataZZ(x) for x in axs]
                self.bxs = [DataZZ(x) for x in bxs]
            else:
                for i in range(len(ciphs)):
                    axs = [c.ax for c in ciphs[i]]
                    bxs = [c.bx for c in ciphs[i]]
                    set_axs = set(axs)
                    fixed_ax_flag = True if len(set_axs)==1 else False
                    self.axs[i] = [DataZZ(axs[0])] if fixed_ax_flag else [DataZZ(x) for x in axs]
                    self.bxs[i] = [DataZZ(x) for x in bxs]

        if print_flag:
            # print('      fixed_ax_flag:', fixed_ax_flag)
            print('  >>> Ciphertexts.shape is '+str(shape)+', using '
                + str(round(time.time()-begin_time, 4)) + ' seconds')


def DataCiphModel(ciph_model, print_flag=False):
    """ transfer model ciphertexts to DataCiphertexts. """
    ciph_data = None
    if type(ciph_model) is type(dict) or type(ciph_model) is type(defaultdict):
        ciph_data = {}
        # print('ciph_model:', ciph_model)
        for k,ciphs in ciph_model:
            ciph_data[k] = DataCiphertexts(ciphs, print_flag=print_flag)
    else:
        ciph_data = DataCiphertexts(ciph_model, print_flag=print_flag)
    return ciph_data


class DataSecretKey():
    """An instance of SecretKey for saving."""
    def __init__(self, sk):
        """ Inits DataSecretKey with the given SecretKey. """
        assert isinstance(sk, SecretKey)
        self.sx = np.array([ int(sk.sx[i]) for i in range(N) ])

class DataKey():
    """An instance of Key for saving."""
    def __init__(self, key):
        """ Inits DataKey with the given Key. """
        assert isinstance(key, Key)
        self.rax = np.array([ int(key.rax[i]) for i in range(Nnprimes) ])
        self.rbx = np.array([ int(key.rbx[i]) for i in range(Nnprimes) ])

def DataKeyShare(ss):
    """ transfer Key Share to data for saving."""
    if isinstance(ss, Uint64_t):
        return [int(ss[i]) for i in range(ss.size())]
    else:
        return ss

def DataKeyShares(keyshares):
    """ transfer Key Shares to data for saving."""
    assert isinstance(keyshares, list) 
    data_keyshares = []
    for ss in keyshares:
        data_keyshares.append( DataKeyShare(ss) )
    return data_keyshares 

def DataZZ(zz, size=None):
    """ transfer ZZ to list. """
    if not size:
        size = N
    if isinstance(zz, ZZ):
        data_zz = [ int(zz[i]) for i in range(size) ]
    elif isinstance(zz[0], ZZ):
        data_zz = []
        for z in zz:
            data_zz.append( [ int(z[i]) for i in range(size) ] )
    else:
        raise ValueError
    return data_zz

def DataUint64_t(uu, size=None):
    """ transfer Uint64_t to list. """
    if isinstance(uu, Uint64_t):
        size = size if size else uu.size()
        data_u = [ int(uu[i]) for i in range(size) ]
    elif isinstance(uu[0], Uint64_t):
        data_u = []
        size = size if size else uu[0].size()
        for u in uu:
            data_u.append( [ int(u[i]) for i in range(size) ] )
    else:
        raise ValueError
    return data_u

class DataLHSSKey():
    """An instance of LHSSKey for saving."""
    def __init__(self, sk):
        """ Inits DataKey with the given Key. """
        assert isinstance(sk, LHSSKey)
        self.id = sk.id
        self.sk = sk.getKeyStr()

class DataLHSPKey():
    """An instance of LHSPKey for saving."""
    def __init__(self, pk):
        """ Inits DataKey with the given Key. """
        assert isinstance(pk, LHSPKey)
        self.id = pk.id
        self.pk = pk.getKeyStr()

class DataLHSSigns():
    """An instance of LHSSigns for saving."""
    def __init__(self, sign):
        """ Inits DataSign with the given LHSSigns. """
        assert isinstance(sign, LHSSigns)
        self.size = sign.size
        self.sign = sign.getSignStr()

#### ------------------------------------------------------------ 

#### ----------------- TRANSFER DATA TO OBJECT ------------------
def dataCiphs2Ciphs(data, print_flag=False):
    ''' transfer DataCiphertexts to a list of Ciphertexts '''
    if isinstance(data, Ciphertexts) or isinstance(data, Ciphertext):
        return data

    if print_flag:
        print('  >>> transfer DataCiphertexts to Ciphertexts ...')
    begin_time = time.time()
    assert isinstance(data, DataCiphertexts)
    logp, logq, n = data.logp, data.logq, data.n
    
    if 'size' in dir(data): # return Class.Ciphertexts
        shape = data.size
        ciphs = Ciphertexts(shape,logp,logq,n,ZZ(data.ax),ZZ(data.bxs))
    else: # return a list of Class.Ciphertext
        shape = data.shape
        if len(shape) == 1:
            axs, bxs = data.axs, data.bxs
            fixed_ax_flag = True if len(axs) == 1 else False
            bx = [ZZ(x) for x in bxs]
            len_cs = len(bx)
            ax = [ZZ(data.axs[0])]*len_cs if fixed_ax_flag else [ZZ(x) for x in axs]
            ciphs = [ Ciphertext(logp,logq,n,ax[i],bx[i]) for i in range(len_cs) ]
        else:
            ciphs = []
            for i in range(len(data.bxs)):
                axs, bxs = data.axs[i], data.bxs[i]
                fixed_ax_flag = True if len(axs) == 1 else False
                bx = [ZZ(x) for x in bxs]
                len_cs = len(bx)
                ax = [ZZ(data.axs[0])]*len_cs if fixed_ax_flag else [ZZ(x) for x in axs]
                ciphs.append([ Ciphertext(logp,logq,n,ax[i],bx[i]) for i in range(len_cs) ])
    if print_flag:
        # print('      fixed_ax_flag:', fixed_ax_flag)
        print('  >>> Ciphertexts.shape is '+str(shape)+', using '
            + str(round(time.time()-begin_time, 4)) + ' seconds')
    return ciphs

def data2Obj(data, class_type, print_flag=False):
    obj = None
    if class_type == 'Ciphertexts':
        obj = dataCiphs2Ciphs(data, print_flag=print_flag)
    elif class_type == 'CiphModel':
        if type(data) is type(dict) or type(data) is type(defaultdict):
            obj = {}
            for k,ciphs in data:
                obj[k] = dataCiphs2Ciphs(ciphs, print_flag=print_flag)
        else:
            obj = dataCiphs2Ciphs(data, print_flag=print_flag)
    elif class_type == 'SecretKey':
        assert isinstance(data, DataSecretKey)
        obj = SecretKey( ZZ(list(data.sx)) )
        #obj.sx.print(2)
    elif class_type == 'Key':
        assert isinstance(data, DataKey)
        obj = Key( Uint64_t(data.rax), Uint64_t(data.rbx) )
    elif class_type == 'ZZ':
        #assert isinstance(data[0], int)
        obj = ZZ(list(data))
    elif class_type == 'Uint64_t':
        #assert isinstance(data[0], int)
        obj = Uint64_t(data)
    elif class_type == 'KeyShare':
        if isinstance(data[0], list):
            obj = [ Uint64_t(data_i) for data_i in data]
        else:
            obj = Uint64_t(data)
    elif class_type == 'KeyShares':
        #assert isinstance(data[0][0], int)
        obj = []
        for data_i in data:
            obj_i = Uint64_t(data_i) if data_i else None
            obj.append(obj_i)
    elif class_type == 'LHSSKey':
        assert isinstance(data, DataLHSSKey)
        obj = LHSSKey()
        obj.setID(data.id)
        obj.setKey(data.sk)
    elif class_type == 'LHSPKey':
        assert isinstance(data, DataLHSPKey)
        obj = LHSPKey()
        obj.setID(data.id)
        obj.setKey(data.pk)
    elif class_type == 'LHSSigns':
        assert isinstance(data, DataLHSSigns)
        obj = LHSSigns(data.size)
        obj.setSign(data.sign)
    else:
        raise ValueError("Not support class_type: {}!".format(class_type))
    return obj
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 


#### ------------------------------------------------------------
#### --------------- MODEL LINEAR HOMORPHIC HASH ----------------
#### ------------------------------------------------------------
def getModelSolnHash(model, soln, msg_len=256):
    """ Get linear homomorphic hash of specific soln of model """
    msg_len = min(len(soln),msg_len)
    lhh_scheme = LHH_init(msg_len=msg_len)

    batch_size = ceil(len(soln)/msg_len)
    soln_mat = soln + [0]*(msg_len*batch_size-len(soln))
    soln_mat = np.array(soln_mat).reshape(-1, batch_size)
    soln_int = [ round(sum(row)) for row in soln_mat]
    # print('soln:', soln[:10])
    # print('soln_int:', soln_int[:10])

    soln_msg = LHHVal(msg_len)
    for i in range(msg_len):
        soln_msg.setVal(str(soln_int[i]), i)

    soln_hash = LHHHash()
    lhh_scheme.getHash(soln_hash, soln_msg)
    return soln_hash
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 


#### ------------------------------------------------------------
#### --------------- CIPHERTEXTS & CIPHERTEXT(S) ----------------
#### ------------------------------------------------------------
def getCiphsAXs(ciphs, eval_time_flag=False):
    if isinstance(ciphs, str):
        ciphs = getObjFromFile(ciphs, 'Ciphertexts')

    if isinstance(ciphs, Ciphertexts):
        ciph_axs = [ ciphs.ax ] * ciphs.size
    elif isinstance(ciphs, list):
        if isinstance(ciphs[0], str):
            ciphs = [ getObjFromFile(ciph,'Ciphertexts') for ciph in ciphs ]
        assert isinstance(ciphs[0], Ciphertext) or isinstance(ciphs[0], Ciphertexts)
        ciph_axs = [ ciph.ax for ciph in ciphs ]
    elif isinstance(ciphs, Ciphertext):
        ciph_axs = ciphs.ax
    else:
        raise ValueError("Input of getCiphsAXs() must be Ciphertext(s) or Ciphertexts(s).")
    return ciph_axs

def getCiphsBX(ciphs, i, eval_time_flag=False):
    if isinstance(ciphs, str):
        ciphs = getObjFromFile(ciphs, 'Ciphertexts')

    if isinstance(ciphs, Ciphertexts):
        ciph_bx = ciphs.getBxs(i)
    elif isinstance(ciphs, list):
        ciph_bx = ciphs[i].bx
    else:
        raise ValueError("Input of getCiphsAXs() must be a Ciphertexts or a list of Ciphertext.")
    return ciph_bx

def getCiphsParams(ciphs, eval_time_flag=False):
    if isinstance(ciphs, str):
        ciphs = getObjFromFile(ciphs, 'Ciphertexts')

    if isinstance(ciphs, list):
        ciphs_num = len(ciphs)
        ciph = ciphs[0]
        if isinstance(ciph, str):
            ciph = getObjFromFile(ciph,'Ciphertexts')
        # assert isinstance(ciph, Ciphertext) 
        logp, logq, n = ciph.logp, ciph.logq, ciph.n
    elif isinstance(ciphs, Ciphertexts):
        ciphs_num = ciphs.size
        logp, logq, n = ciphs.logp, ciphs.logq, ciphs.n
    else:
        raise ValueError("Input of getCiphsAXs() must be a Ciphertexts or a list of Ciphertext.")
    return ciphs_num, logp, logq, n

def lenCiphs(ciphs):
    if isinstance(ciphs, str):
        ciphs = getObjFromFile(ciphs, 'Ciphertexts')

    if isinstance(ciphs, list):
        ciphs_num = len(ciph_sum)
    elif isinstance(ciphs, Ciphertexts):
        ciphs_num = ciphs.size
    else:
        raise ValueError("Input of getCiphsAXs() must be a Ciphertexts or a list of Ciphertext.")
    return ciphs_num
