import random
from math import ceil, floor
import torch
import numpy

def evaluation_model(soln, eval_method='l2', eval_data=None):
    ''' (correctness test) the real evaluation result of local model
        Args:
            soln: a local model (torch)
            eval_method:
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
    '''
    print('(Correctness Test) Client correct evaluation results:')
    real_eva_res = 0
    soln = torch.tensor(soln)

    if eval_method == 'l2': # no eva_data
        real_eva_res = torch.norm(soln, p=2)

    elif eval_method == 'ln': # eval_data: (valid_soln, ciph_valid_soln)
        vaild_soln,_ = eval_data
        vaild_soln = torch.tensor(vaild_soln)
        real_eva_res = torch.dist(soln, vaild_soln, p=2)

    elif eval_method == 'zeno': 
        vaild_soln,a,b,_ = eval_data
        vaild_soln = torch.tensor(vaild_soln)
        real_eva_res = a * torch.dot(soln, vaild_soln)
        real_eva_res -= b * torch.norm(vaild_soln, p=2)

    elif eval_method == 'cos': # eva_data = gm_soln
        vaild_soln,_ = eval_data
        vaild_soln = torch.tensor(vaild_soln)
        real_eva_res = torch.dot(soln, vaild_soln)
        real_eva_res /= torch.norm(vaild_soln, p=2)
        # print('cos:', torch.cosine_similarity(soln, vaild_soln, dim=0))
    else:
        raise ValueError("Not support evaluation method: {}!".format(eval_method))
    
    return real_eva_res




