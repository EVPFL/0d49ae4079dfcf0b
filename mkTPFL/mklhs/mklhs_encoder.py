"""A module to encode integers as specified in the MKCKKS scheme.
"""
from MKLHS import LHSMsgs, MKLHSMsgs
from mkTPFL.mklhs.mklhs_parameters import *

from HEAAN import ZZ, Plaintext, Ciphertext, Ciphertexts
from mkTPFL.ckks.ckks_evaluator import CKKSEvaluator
from mkTPFL.ckks.util import getZZMAC, getPlaintxtMAC, getCiphtxtMAC

from LHH import LHHHash
from mkTPFL.utils.lhh_parameters import LHH_init

from tqdm import tqdm


class MKLHSEncoder():

    """ An object that can encode data for MKLHS scheme. 
    """

    def __init__(self):
        """Generates a encoder for MKLHS scheme.
        """
        global mklhs_scheme
        if not mklhs_scheme:
            mklhs_scheme = MKLHS_init()
        self.params = mklhs_scheme.params


    def __encode__(self, m, mod=None):
        """ transfer input to the str(hex) """
        m_str, mod_str = "", ""
        if isinstance(m, int):
            m_str = str(hex(m))[2:]

        elif isinstance(m, str):
            if m[0:2] == "0x" or m[0:2] == "0X":
                m_str = m_str[2:]
            else:
                try:
                    m_str = str(int(m))[2:]
                except:
                    m_str = "0"

        elif isinstance(m, ZZ):
            mac_m, mac_mod = getZZMAC(m, mod)
            m_str = str(hex(mac_m))[2:]
            mod_str = str(hex(mac_mod))[2:]

        elif isinstance(m, Plaintext):
            mac_m, mac_mod = getPlaintxtMAC(m)
            m_str = str(hex(mac_m))[2:]
            mod_str = str(hex(mac_mod))[2:]

        elif isinstance(m, Ciphertext):
            mac_m, mac_mod = getCiphtxtMAC(m)
            m_str = str(hex(mac_m))[2:]
            mod_str = str(hex(mac_mod))[2:]

        elif isinstance(m, LHHHash):
            lhh_scheme = LHH_init()
            m_str = lhh_scheme.getHashesHex(m)
            # mac_mod = 2*1000
            # mod_str = str(hex(mac_mod))[2:]

        else:
            raise ValueError("Input to encode must be int(s), str(s) which can transfer to int, or Ciphertext(s) of CKKS.")

        return m_str, mod_str

    def encodeLHSMsgs(self, msgs, mod=None, msgs_num=0):
        """ """
        if isinstance(msgs, LHSMsgs):
            return msgs

        msgs_num = max(2, msgs_num)
        msg_list = msgs if isinstance(msgs, list) else [msgs]
        if len(msg_list) < msgs_num:
            msg_list = msg_list + [0]*(msgs_num-len(msg_list))
        
        lhsmsg = LHSMsgs(msgs_num)
        for i in range(msgs_num):
            m_str, mod_str = self.__encode__(msg_list[i], mod=mod)
            mod_str = mod_str if mod_str else self.params.getNStr()
            lhsmsg.setMsg(m_str, i, mod_str)
        
        return lhsmsg

    def encodeMKLHSMsgs(self, lhs_msgs, signer_num=None, msgs_num=None):
        """ """
        if isinstance(lhs_msgs, LHSMsgs):
            signer_num = 1
            msgs_num = lhs_msgs.size
            lhs_msgs = [lhs_msgs]

        if isinstance(lhs_msgs, list):
            signer_num = len(lhs_msgs)
            for i in range(signer_num):
                if not isinstance(lhs_msgs[i], LHSMsgs):
                    lhs_msgs[i] = self.encodeLHSMsgs(lhs_msgs[i])
            msgs_nums = [m.size for m in lhs_msgs]
            assert len(set(msgs_nums))==1
            msgs_num = msgs_nums[0]

        mklhsmsg = MKLHSMsgs(signer_num, msgs_num)
        for i in range(signer_num):
            mklhsmsg.setMsgs(lhs_msgs[i], i)
        
        return mklhsmsg

    def ciphs2LHSMsgs(self, ciphs):
        """ transfer Ciphstexts to the LHSMsgs """
        if isinstance(ciphs, Ciphertext):
            ciphs = [ciphs]
        evaluator = CKKSEvaluator()
        ciph_sum = evaluator.sum(ciphs)
        lhsmsg = self.encodeLHSMsgs(ciph_sum)
        return lhsmsg


    def __hexstr2Int__(self, m_str, mod):
        """ transfer str(hex) to int """
        assert not mod or isinstance(mod, int)

        if not m_str:
            m = 0
        elif isinstance(m_str, int):
            m = m_str
        elif isinstance(m_str, str):
            try:
                m = int(m_str, 16)
            except:
                raise ValueError("Input must be int/str(hex) which can transfer to a int.")
        return m if not mod else m%mod


    def hexstrs2Ints(self, m_strs, mod=None, mod_flag=True):
        """ transfer str(hex)(s) to int(s) """
        mod = self.__hexstr2Int__(mod, None)  
        if mod_flag and not mod:
            mod_str = self.scheme.params.getNStr()
            mod = int(mod, 16)

        if isinstance(m_strs, int) or isinstance(m_strs, str):
            return self.__hexstr2Int__(m_strs, mod)
        elif isinstance(m_strs, list):
            return [ self.__hexstr2Int__(m_str, mod) for m_str in m_strs ]
        else:
            raise ValueError("Input must be int/str(hex) which can transfer to int(s).")


    def __lhsmsgs2Ints__(self, msgs):
        """ transfer LHSMsgs to list(int) (i.e., one-dimensional array) """
        assert isinstance(msgs, LHSMsgs)
        msgs_str_list = msgs.getMsgsStr().split('-')
        while '' in msgs_str_list:
            msgs_str_list.remove('')
        msg_ints = [ int(m_str, 16) for m_str in msgs_str_list ]
        return msg_ints

    def msgs2Ints(self, msgs):
        """ transfer LHSMsgs/MKLHSMsgs to list(int) (i.e., one/two-dimensional array) """
        msgs_ints = []
        if isinstance(msgs, LHSMsgs):
            return self.__lhsmsgs2Ints__(msgs)
        elif isinstance(msgs, MKLHSMsgs):
            msgs = [ msgs[i] for i in range(msgs.signers_num) ]

        if isinstance(msgs, list):
            for msg in msgs:
                msgs_ints.append( self.__lhsmsgs2Ints__(msg) )
        else:
            raise ValueError("Input must be a LHSMsgs(s) or MKLHSMessage.")
        return msgs_ints


    def __lhslfs2Ints__(self, f):
        """ transfer LHSLinearity to list(int) (i.e., one-dimensional array) """
        f_ints = []
        if isinstance(f, LHSLinearity):
            f_ints = [ f[i] for i in range(f.size) ]
        elif isinstance(f, int):
            f_ints = f
        elif isinstance(f, list):
            try:
                f_ints = [ int(f_i) for f_i in f ]
            except:
                ValueError("Input must be a LHSLinearity or int(s).")

        return f_ints

    def lfs2Ints(self, fs):
        """ transfer LHSLinearity/MKLHSLinearity to list(int) (i.e., one/two-dimensional array) """
        fs_ints = []
        if isinstance(fs, LHSLinearity):
            fs_ints = self.__lhslfs2Ints__(fs)
        elif isinstance(fs, MKLHSLinearity):
            fs = [ fs[i] for i in range(fs.signers_num) ]

        if isinstance(fs, list):
            for f in fs:
                fs_ints.append( self.__lhslfs2Ints__(f) )
        else:
            raise ValueError("Input must be int(s), LHSLinearity(s) or MKLHSLinearity.")
        return fs_ints


    def trans2LHSLinearity(self, weight, msgs_num):
        """ """
        if isinstance(weight, LHSLinearity) and weight.msgs_num==msgs_num:
            return weight

        f = LHSLinearity(msgs_num)
        if not weight:
            f = f
        elif isinstance(weight, LHSLinearity):
            f.copyCoeff(weight)
        elif isinstance(weight, list) and len(weight) == msgs_num:
            for j in range(msgs_num):
                f.setCoeff(weight[j], j)
        else:
            try:
                for j in range(msgs_num):
                    f.setCoeff(weight, j)
            except:
                raise ValueError("Input [f] must be a LHSLinearity(s) or other element(s) which can transfer to LHSLinearity.")
        return f


    def trans2MKLHSLinearity(self, weights, signers_num, msgs_num):
        """ """
        if isinstance(weights, MKLHSLinearity) and weights.signers_num==signers_num and weights.msgs_num==msgs_num:
            return weights

        fs = MKLHSLinearity(signers_num, msgs_num)
        if not weights:
            fs = fs
        elif isinstance(weights, MKLHSLinearity):
            for i in range(signers_num):
                fs[i].copyCoeff(weights[i])
        elif isinstance(weights, list) and len(weights) == signers_num:
            for i in range(signers_num):
                try:
                    f = self.trans2LHSLinearity(weights[i], msgs_num)
                    fs[i].copyCoeff(f)
                except:
                    raise ValueError("Input [fs] must be a MKLHSLinearity, LHSLinearity(s) or other element(s) which can transfer to MKLHSLinearity.")
                
        else:
            raise ValueError("Input [fs] must be a MKLHSLinearity, LHSLinearity(s) or other element(s) which can transfer to MKLHSLinearity.")
        # print('fs:')
        # for i in range(fs.signers_num):
        #     fs[i].print()
        return fs



#### ------------------------------------------------------------
#### -------- MODEL CIPHERTEXT & LHS MESSAGE TRANSFORMER --------
#### ------------------------------------------------------------
def modelCiph2LHSMsg(ciph_model, ckks_params):
    """ transfer CKKS Ciphertxts of model to LHS Message """
    evaluator = CKKSEvaluator(ckks_params)
    encoder = MKLHSEncoder()

    if isinstance(ciph_model, str): # ciph_model is the file path
        data = getData(file_path=ciph_model, print_flag=True)
        ciph_model = data2Obj(data, 'CiphModel', print_flag=True)

    lhsmsg_ciph_model = encoder.ciphs2LHSMsgs(ciph_model)
    return lhsmsg_ciph_model
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 
