import numpy as np
import torch
import random
from tqdm import tqdm

from mkTPFL.ckks.ckks_parameters import *
from mkTPFL.ckks.ckks_encoder import CKKSMessage
from mkTPFL.ckks.ckks_evaluator import CKKSEvaluator
from mkTPFL.ckks.ckks_encryptor import CKKSEncryptor
from HEAAN import Ciphertext, Ciphertexts, Double, ComplexDouble

from mkTPFL.ckks.ckks_decryptor import CKKSDecryptor



class EncModelEvaluator():
    ''' Evaluator for encrypted local model (enc_lm: Ciphtexts)
       
        eval_method: (corresponding to mkTPFL.utils.test_util.evaluation_model)
            - 'l2': 
                eva_data: None
                output: torch.norm(soln, p=2)
            - 'ln': 
                eva_data: vaild_soln (train on public dataset)
                output: torch.dist(soln, eva_data, p=2)
            - 'zeno':
                eva_data: (vaild_soln,a,b), where a,b is threshold
                output: a*torch.dot(soln, vaild_soln)-b*torch.norm(vaild_soln, p=2)
            - 'cos': 
                eva_data: gm_soln (lastest global model)
                output: torch.dot(soln, gm_soln)/torch.norm(gm_soln, p=2)
        Args:
            - ckks_params
            - pk: public key of enc_lm
            - rk: relin key of enc_lm
            - soln_scaling: the rescale paramater using during encryption of local updates (local_soln).
                i.e., enc_lm = Enc(lm), where lm = [round(float(v)*soln_scaling) for v in local_soln]

    '''

    def __init__(self, ckks_params, pk, rk, eval_method='l2', soln_scaling=10**6):
        self.ckks_params = ckks_params
        self.evaluator = CKKSEvaluator(ckks_params)
        self.pk = pk
        self.rk = rk 
        self.eval_method = eval_method
        self.scaling = Double( 1 / soln_scaling )


    def eval(self, ciph_model, eval_data=None, eval_method=None):
        ''' evaluate encrypted local model by eval_method
            output:
                ciph_eval (Ciphertext) = eval_method(ciph_model)
        '''
        eval_method = eval_method if eval_method else self.eval_method

        evaluator = self.evaluator
        pk, rk = self.pk, self.rk
        scaling = self.scaling
        
        ciph_eval = None
        if eval_method == 'l2': 
            ciph_eval = eval_l2(evaluator, pk, rk, scaling, ciph_model)
        elif eval_method == 'ln':
            ciph_eval = eval_ln(evaluator, pk, rk, scaling, ciph_model, eval_data)
        elif eval_method == 'zeno':
            ciph_eval = eval_zeno(evaluator, pk, rk, scaling, ciph_model, eval_data)
        elif eval_method == 'cos': 
            ciph_eval = eval_cos(evaluator, pk, rk, scaling, ciph_model, eval_data)
        else:
            raise ValueError("Not support evaluation method: {}!".format(eval_method))
        return ciph_eval


    def preprocessing(self, worker, verify_dataloader, eval_method=None, valid_soln=None, ciph_model=None):
        ''' get the eva_data for self.eval() based on eval_method
            output: eva_data
        '''
        eval_method = eval_method if eval_method else self.eval_method
        ckks_params, pk = self.ckks_params, self.pk
        soln_scaling = 1 / float(str(self.scaling[0]))
        eval_data = preprocessing(worker, verify_dataloader, eval_method=eval_method, 
            valid_soln=valid_soln, ckks_params=ckks_params, pk=pk, ciph_model=ciph_model, soln_scaling=soln_scaling)
        return eval_data


    def getMask(self, eval_method=None):
        ''' offline '''
        n, logp, logq = self.ckks_params.n, self.ckks_params.logp, self.ckks_params.logq
        eval_method = eval_method if eval_method else self.eval_method
        p = int(ring.qpows[logp])
        soln_scaling = int(1 / float(str(self.scaling[0])))

        if eval_method in ['l2', 'ln', 'cos', 'zeno']: 
            logp *= 4
        else:
            raise ValueError("Not support evaluation method: {}!".format(eval_method))
            
        r1 = random.randint(1, p-1)
        r2 = [ random.randint(1, soln_scaling) for _ in range(self.ckks_params.slots) ]
        r3 = [ random.randint(1, soln_scaling) for _ in range(self.ckks_params.slots) ]
        msg_r2 = CKKSMessage(slots=self.ckks_params.slots, mlist=r2)
        msg_r3 = CKKSMessage(slots=self.ckks_params.slots, mlist=r3)
        encryptor = CKKSEncryptor(self.ckks_params, self.pk)
        ciph_r2 = encryptor.encrypt(msg_r2, n=n, logp=logp, logq=logq)
        ciph_r3 = encryptor.encrypt(msg_r3, n=n, logp=logp*2, logq=logq)
        return r1, r2, r3, ciph_r2, ciph_r3


    def mask(self, ciph_eva, mask_data):
        ''' online '''
        r1, _, _, ciph_r2, ciph_r3 = mask_data
        # assert ciph_r2.n == ciph_eva.n and ciph_r2.logp == ciph_eva.logp and ciph_r2.logq == ciph_eva.logq
        self.evaluator.addAndEqual(ciph_r2, ciph_eva)
        self.evaluator.multByConstAndEqual(ciph_r2, Double(r1), ciph_r2.logp)
        # assert ciph_r2.n == ciph_r3.n and ciph_r2.logp == ciph_r3.logp and ciph_r2.logq == ciph_r3.logq
        self.evaluator.addAndEqual(ciph_r2, ciph_r3)
        return ciph_r2


    def unmask(self, sum_mask_eval, mask_data, eval_method=None, eval_data=None):
        ''' online '''
        eval_method = eval_method if eval_method else self.eval_method
        return unmask(sum_mask_eval, mask_data, eval_method=eval_method, eval_data=eval_data)

    
def unmask(sum_mask_eval, mask_data, eval_method='l2', eval_data=None):
    ''' online '''
    r1, r2, r3, _, _ = mask_data
    # eval_res = sum_mask_eval - sum(r2)
    # eval_res /= r1
    eval_res = sum_mask_eval - sum(r3)
    eval_res /= r1
    eval_res -= sum(r2)
    if eval_method == 'l2' or eval_method == 'ln': 
        return pow(eval_res, 0.5)
    elif eval_method == 'zeno': 
        vaild_soln,a,b,_ = eval_data
        return (a*eval_res) - (b*torch.norm(vaild_soln, p=2))
    elif eval_method == 'cos': 
        vaild_soln,_ = eval_data
        return eval_res / torch.norm(vaild_soln, p=2)
    else:
        raise ValueError("Not support evaluation method: {}!".format(eval_method))


def preprocessing(worker, verify_dataloader, eval_method='l2', valid_soln=None, ckks_params=None, pk=None, ciph_model=None, soln_scaling=10**6):
    ''' offline '''
    eval_data = None

    if eval_method == 'l2': 
        eval_data = None

    elif eval_method == 'ln':
        if isinstance(valid_soln,type(None)):
            valid_soln,_ = worker.local_train(verify_dataloader)
        encryptor = CKKSEncryptor(ckks_params, pk)
        n,logp,logq = ciph_model.n, ciph_model.logp, ciph_model.logq
        # scaling_soln = [ -x*soln_scaling for x in valid_soln ]
        scaling_soln = list(-valid_soln*soln_scaling)
        msg_soln = CKKSMessage(slots=ckks_params.slots, mlist=scaling_soln)
        ciph_soln = encryptor.encrypt(msg_soln, thread_num=8, list_flag=False, n=n, logp=logp, logq=logq)
        eval_data = valid_soln, ciph_soln

    elif eval_method == 'cos' or eval_method == 'zeno': 
        if isinstance(ciph_model, Ciphertexts):
            ciph_model_size = ciph_model.size
        else:
            ciph_model_size = 1
            ciph_model = [ciph_model]

        # assert valid_soln
        slots = ckks_params.slots
        padding_num = ciph_model_size*slots-len(valid_soln)
        valid_soln_list = list(valid_soln) + [0]*padding_num
        valid_soln_vec = []
        for i in range(ciph_model_size):
            tmp = ComplexDouble( valid_soln_list[i*slots:i*slots+slots] )
            valid_soln_vec.append(tmp)

        if eval_method == 'cos':
            eval_data = valid_soln, valid_soln_vec

        else: # eval_method == 'zeno': 
            a,b = 1,0
            eval_data = valid_soln, a, b, valid_soln_vec
    else:
        raise ValueError("Not support evaluation method: {}!".format(eval_method))

    return eval_data



def eval_l2(evaluator, pk, rk, scaling, ciph_model):
    # assert isinstance(ciph_model, Ciphertexts)
    if isinstance(ciph_model, Ciphertexts):
        ciph_model_size = ciph_model.size
    else:
        ciph_model_size = 1
        ciph_model = [ciph_model]

    ciph_eva = None
    for i in tqdm( range(ciph_model_size) ):
        ciph_tmp = Ciphertext()
        evaluator.multByConst(ciph_tmp, ciph_model[i], scaling, ciph_model[i].logp)
        evaluator.squareAndEqual(ciph_tmp, rk)
        if not ciph_eva:
            ciph_eva = ciph_tmp
        else:
            evaluator.addAndEqual(ciph_eva, ciph_tmp)
    return ciph_eva


def eval_ln(evaluator, pk, rk, scaling, ciph_model, data):
    # assert isinstance(ciph_model, Ciphertexts)
    _, ciph_vmodel = data
    if isinstance(ciph_model, Ciphertexts):
        ciph_model_size = ciph_model.size
    else:
        ciph_model_size = 1
        ciph_model = [ciph_model]
        ciph_vmodel = [ciph_vmodel]

    ciph_eva = None
    for i in tqdm( range(ciph_model_size) ):
        ciph_tmp = Ciphertext()
        evaluator.add(ciph_tmp, ciph_model[i], ciph_vmodel[i])
        evaluator.multByConstAndEqual(ciph_tmp, scaling, ciph_tmp.logp)
        evaluator.squareAndEqual(ciph_tmp, rk)
        if not ciph_eva:
            ciph_eva = ciph_tmp
        else:
            evaluator.addAndEqual(ciph_eva, ciph_tmp)
    return ciph_eva


def eval_zeno(evaluator, pk, rk, scaling, ciph_model, data):
    # assert isinstance(ciph_model, Ciphertexts)
    if isinstance(ciph_model, Ciphertexts):
        ciph_model_size = ciph_model.size
    else:
        ciph_model_size = 1
        ciph_model = [ciph_model]

    _,_,_, vmodel = data
    ciph_eva = None
    for i in tqdm( range(ciph_model_size) ):
        ciph_tmp = Ciphertext()
        evaluator.multByConst(ciph_tmp, ciph_model[i], scaling, ciph_model[i].logp)
        evaluator.multByConstVecAndEqual(ciph_tmp, vmodel[i], ciph_tmp.logp)
        if not ciph_eva:
            ciph_eva = ciph_tmp
        else:
            evaluator.addAndEqual(ciph_eva, ciph_tmp)

    return ciph_eva


def eval_cos(evaluator, pk, rk, scaling, ciph_model, data):
    # assert isinstance(ciph_model, Ciphertexts)
    if isinstance(ciph_model, Ciphertexts):
        ciph_model_size = ciph_model.size
    else:
        ciph_model_size = 1
        ciph_model = [ciph_model]

    _, vmodel = data
    ciph_eva = None
    for i in tqdm( range(ciph_model_size) ):
        ciph_tmp = Ciphertext()
        evaluator.multByConst(ciph_tmp, ciph_model[i], scaling, ciph_model[i].logp)
        evaluator.multByConstVecAndEqual(ciph_tmp, vmodel[i], ciph_tmp.logp)
        if not ciph_eva:
            ciph_eva = ciph_tmp
        else:
            evaluator.addAndEqual(ciph_eva, ciph_tmp)

    return ciph_eva





