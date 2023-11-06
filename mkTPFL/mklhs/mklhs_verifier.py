"""A module to verify the signature in the MKCKKS scheme.
"""
from MKLHS import *
from mkTPFL.mklhs.mklhs_parameters import *
from mkTPFL.mklhs.mklhs_encoder import *

import numpy as np

class MKLHSVerifier():

    """ An object that can verify signature for MKLHS scheme. 
    """

    def __init__(self):
        """Generates a encoder for MKLHS scheme.
        """
        global mklhs_scheme
        if not mklhs_scheme:
            mklhs_scheme = MKLHS_init()
        self.scheme = mklhs_scheme

    def isMsgLinearity(self, msg_sum, msgs, weights=None, mod=None):
        """ msg_sum == sum(msgs)%mod """
        encoder = MKLHSEncoder()
        msg_sum_int = encoder.msgs2Ints(msg_sum)
        msgs_int = np.array(encoder.msgs2Ints(msgs))

        if not weights:
            fs_int = np.ones(msgs_int.shape)
        else:
            fs_int = np.array(encoder.lfs2Ints(weights))

        if msgs_int.shape != fs_int.shape:
            assert len(msgs_int) == len(fs_int) and len(fs_int.shape) == 1
            f_len = msgs_int.shape[1]
            fs_int = np.array( [ [fs_int[i]]*f_len for i in range(len(fs_int)) ] )

        if not mod:
            mod_str = self.scheme.params.getNStr()
            mod = int(mod_str, 16)
        else:
            mod = encoder.hexstrs2Ints(mod, mod_flag=False)
        # assert isinstance(mod, int)

        for i in range(len(msg_sum_int)):
            res_sum = sum([ int(msgs_int[j][i])*int(fs_int[j][i]) for j in range(len(msgs_int)) ])
            if int( (msg_sum_int[i]-int(res_sum)) %mod) != 0:
                print( 'res_dif[{0}]: {1}'.format( i, int( (msg_sum_int[i]-int(res_sum)) %mod)) )
                return False
        return True


    def verifyLHSign(self, msg, sign, pk, weight=None):
        """ verify the signature of one singer """
        assert(sign, LHSSigns)
        encoder = MKLHSEncoder()

        if not isinstance(msg, LHSMsgs):
            msg = encoder.encodeLHSMsgs(msg)
        signers_num, msgs_num = 1, msg.size

        f = encoder.trans2LHSLinearity(weight, msgs_num)

        sign_res = LHSSigns()
        self.scheme.evalSign(sign_res, sign, f)

        mkpks = MKLHSPKeys(signers_num)
        mkpks.setPKs(pk, 0)

        mkmsgs = MKLHSMsgs(signers_num, msgs_num)
        mkmsgs.setMsgs(msg, 0)
        fs = MKLHSLinearity(signers_num, msgs_num)
        fs.setCoeffs(f, 0)
        ver_res = self.scheme.verMKLHS(sign_res, mkmsgs, fs, mkpks)

        if ver_res == 1:
            return True
        else:
            return False


    def verifyMKLHSign(self, msgs, sign_res, pks, weights=None):
        """ verify the signatures of mutiple singers """
        assert(sign_res, LHSSigns)
        
        encoder = MKLHSEncoder()
        if isinstance(msgs, list) or isinstance(msgs, LHSMsgs):
            msgs = encoder.encodeMKLHSMsgs(msgs)
        elif not isinstance(msgs, MKLHSMsgs):
            raise ValueError("Input [msgs] must be a MKLHSMsgs, LHSMsgs(s) or other element(s) which can transfer to MKLHSMsgs.")
        mkmsgs = msgs
        signers_num, msgs_num = msgs.signers_num, msgs.msgs_num
        
        fs = encoder.trans2MKLHSLinearity(weights, signers_num, msgs_num)
        
        if isinstance(pks, list) and len(pks) == signers_num:
            mkpks = MKLHSPKeys(signers_num)
            for i in range(signers_num):
                mkpks.setPKs(pks[i], i)
            pks = mkpks
        elif not isinstance(pks, MKLHSPKeys):
            raise ValueError("Input [pks] must be a MKLHSPKeys or LHSPKey(s).")
        if pks.signers_num != signers_num:
            raise ValueError("Input [pks]'s signers_num must be equal to the msgs.")
        mkpks = pks

        ver_res = self.scheme.verMKLHS(sign_res, mkmsgs, fs, mkpks)
        if ver_res == 1:
            return True
        else:
            return False


