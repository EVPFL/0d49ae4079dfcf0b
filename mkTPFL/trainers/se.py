from mkTPFL.ckks.ckks_parameters import * # for test
from mkTPFL.trainers.base import BaseMKTPFLTrainer

import time
import torch
from collections import defaultdict
import numpy as np
import math

from mkTPFL.utils.class_transformer import *
from mkTPFL.utils.test_util import evaluation_model


class SecEvalTester(BaseMKTPFLTrainer):
    def __init__(self, options, dataset, verify_data=None):
        dataset_name = options['dataset']
        pfl_options = defaultdict(lambda:False)

        # Set the evaluation method
        pfl_options['eval_client'] = True
        # pfl_options['eval_method'] = options['eval_method']
    
        # accuracy of input vectors
        pfl_options['soln_scaling'] = 10**6
        # drop-out simulation
        if 'dropout_ratio' not in options:
            options['dropout_ratio'] = 0.5

        options.update(pfl_options)
        super(SecEvalTester, self).__init__(options, dataset, verify_data=verify_data)
        # print('stat_dir:', self.stat_dir)


    def local_input_solns(self, selected_clients, seed=1):
        """ solns = [ (weight, soln), ... ]
        """
        solns = []
        for c in selected_clients:
            np.random.seed(seed+c.cid)
            rv = np.random.rand(self.input_length)
            rv = np.array([ round(x, 4) for x in rv ])
            solns.append( (c.weight, rv) )
        return solns


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Generate the aggregation public key
        self.set_agg_pk()

        for round_i in range(self.num_round):
            print('======================================================================================================')
            print('ROUND [{}]'.format(round_i))

            # Choose 1 clients to do the secure evaluation
            selected_clients = self.select_clients(seed=self.seed+round_i)[:1]

            # Input locally
            if self.dataset_name not in ['randomvector','rv']: # real model
                self.latest_model = self.worker.get_flat_model_params().detach()
                solns, stats = self.local_train(round_i, selected_clients)
            else: # random_vector
                self.worker = None
                self.latest_model = np.array([ round(x, 4) for x in np.random.rand(self.input_length) ])
                solns = self.local_input_solns(selected_clients, seed=self.seed+round_i)

            # # [selected DI] Encrypt & Sign locally
            # partEncrypt(enc_serect_key) & sign(partciph_s)
            partciph_s_dict, sign_cps_dict, local_stats = self.local_encrypt_iterkey(selected_clients, save_flag=False, round_num=round_i)
            # signModel(lm), encrypt(lm) & sign(ciph_lm)
            sign_lm_dict, ciph_lm_dict, local_lm_stats = self.local_encrypt_soln(selected_clients, solns, save_flag=False, round_num=round_i)

            # evaluate the local models ciphertexts
            cid, soln = selected_clients[0].cid, solns[0][1]
            ciph_lm = ciph_lm_dict[cid]
            print('  eval client:', cid)
            eva_res, eva_data, sv_stats, di_stats =  self.evaluation_localmodel(cid, ciph_lm)
            print('  eval_res:', eva_res)
            
            # correctness test
            eval_res_cor = evaluation_model(soln, eval_method=self.eval_method, eval_data=eva_data)
            print('  eval_res_cor:', eval_res_cor)
            eva_res = torch.tensor(eva_res)
            # dis = torch.dist(eva_res, eval_res_cor, p=2)
            dis = 1 - abs(eva_res-eval_res_cor)/eval_res_cor
            print('  distance between correct_eva and compute_eva:', float(dis))

            # Track cost
            stats_dict = {  '[DI] eval_clm ': di_stats,
                            '[SV] eval_clm': sv_stats,
                            'Dataset': self.dataset_name,
                            'Model': self.model_name,
                            'SE_Method': self.eval_method,
                            'SE_ACC': float(dis)}
            # print('stats_dict:', stats_dict)

            # self.log_stats(round_i, selected_clients, stats_dict)
            file_path = mkdir(self.stat_dir)+'se_stats.xlsx'
            self.log_stats_sheet(round_i, stats_dict, file_path=file_path)

            print('\n>>> Round: {} / SE_MSE: {}'.format(round_i, float(dis)) )
            print('======================================================================================================\n')

            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

            