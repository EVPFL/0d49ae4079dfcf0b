from mkTPFL.ckks.ckks_parameters import * # for test
from mkTPFL.trainers.base import BaseMKTPFLTrainer

import time
import torch
from collections import defaultdict
import numpy as np
import math

from mkTPFL.utils.class_transformer import *


class SATrainer(BaseMKTPFLTrainer):
    def __init__(self, options, dataset, verify_data=None):
        dataset_name = options['dataset']

        pfl_options = defaultdict(lambda:False)

        # (SATrainer do not evaluate local models)
        pfl_options['eval_client'] = False 
        # accuracy of input vectors
        pfl_options['soln_scaling'] = 10**6
        # drop-out simulation
        if 'dropout_ratio' not in options:
            options['dropout_ratio'] = 0.5

        options.update(pfl_options)
        super(SATrainer, self).__init__(options, dataset)
        

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


    def aggregate(self, solns):
        """ Aggregate local solutions and output new global parameter
            Args:
                solns: a generator or (list) with element (weight, local_solution)
            Returns:
                flat global model parameter
        """
        averaged_solution = torch.zeros_like( torch.Tensor(solns[0][1]) )
        if self.simple_average:
            num = 0
            for weight, local_solution in solns:
                num += 1
                averaged_solution += local_solution
            averaged_solution /= num
        else:
            weight_sums = 0
            for weight, local_solution in solns:
                averaged_solution += weight * local_solution
                weight_sums += weight
            averaged_solution /= weight_sums
        return averaged_solution.detach()


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Generate the aggregation public key
        if self.dropout_ratio>0:
            # using aggregation public key every round
            self.set_agg_pk()
            global_agg_pk_flag = True
        else: # change aggregation public key each round (baseline)
            global_agg_pk_flag = False

        for round_i in range(self.num_round):
        # for round_i in range(1):
            print('======================================================================================================')
            print('ROUND [{}]'.format(round_i))
            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=self.seed+round_i)

            # Generate the aggregation public key
            if not global_agg_pk_flag: 
                self.set_agg_pk(selected_clients)

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
            for cid in local_stats.keys():
                local_stats[cid].update(local_lm_stats[cid])
            # # get ciph_lm & ciph_s from file
            # print('>>> get partciph_s_dict & ciph_lm_dict from file')
            # partciph_s_dict = self.get_partciph_s_dict_from_file(selected_clients, round_num=round_i)
            # ciph_lm_dict = self.get_ciph_lm_dict_from_file(selected_clients, round_num=round_i)
            
            # # [SV] Verify: verifySign(sign_cps_dict,sign_lm_dict)
            ver_sign_cps_res, ver_local_sign_dict, msg_ciph_s_dict, ver_sign_local_stats = self.verification_localsign(partciph_s_dict, sign_cps_dict, sign_lm_dict, selected_clients=selected_clients) # verify signatures
            # if there is invalid signature of ciph_iterkey, abort the aggregation
            if not ver_sign_cps_res[0]: 
                print('\n>>> Round[{}] failed: client[{}]\'s signature of ciph_iterkey is invalid'.format(round_i, ver_sign_cps_res[1]) )
                print('======================================================================================================\n')
                continue

            # # Simulation: exclude the model-evaluation-failed and drop-out seleted clients
            selected_cids = [c.cid for c in selected_clients]
            failed_clients_ids, dp_clients_ids, excluded_clients_ids = [], [], []
            # randomly select drop-out clients from selected clients in this aggregation
            # print('dropout_ratio:',  self.dropout_ratio)
            if self.dropout_ratio > 0:
                dp_clients_ids = self.simulation_dropout(seed=self.seed+round_i, selected_ids=selected_cids)

            # # [SV] Aggregation iterkey ciphertexts: ciph_s_sum = aggregation(partciph_s) for clients who passed signuature verification & local model evaluation
            ciph_s_sum, sign_ciph_s_sum, agg_ciph_s_stats = self.aggregation_partciph_iterkey(partciph_s_dict, sign_cps_dict)

            # # [online DI] Verify ciph_iterkey & Decrypt shares locally
            # verify signature of ciph_iterkey locally (only simulating DIs[0])
            ver_sign_cps_sum, ver_sign_cps_sum_stats = self.ver_sign_ciph_iterkey(sign_ciph_s_sum, ciph_s_sum, msg_ciph_s_dict)
            if len(dp_clients_ids) == 0: # all selected clients are online (Baseline)
                self.dropout_flag = False
                # [selected DI] calculate partdecshare of ciph_iterkey locally
                # decshare_s_dict, decshare_stats = self.local_decshare(ciph_s_sum, recovered_clients=selected_clients) 
                decshare_s_dict, decshare_stats = self.local_decshare(ciph_s_sum, recovered_clients=selected_clients)
            else: # some selected clients are drop-out
                self.dropout_flag = True
                online_clients = [ c for c in self.clients if c.cid not in dp_clients_ids ]
                # [online DI] calculate partdecshare_shares of ciph_iterkey locally
                decshare_s_dict, decshare_stats = self.local_decshare_sss(ciph_s_sum, online_clients)
            # decshare_stats.update(ver_sign_cps_sum_stats)
            for cid in ver_sign_cps_sum_stats.keys():
                decshare_stats[cid].update(ver_sign_cps_sum_stats[cid])
            
            # # [SV] Get the iterkey by decshare_s_dict: iterkey = decshareMerge(ciph_s_sum, decshare_s_list)
            iterkey, get_iterkey_stats = self.get_iterkey(ciph_s_sum, decshare_s_dict, sss_flag=self.dropout_flag)

            # # [SV] Get & Sign global model
            gm, sign_gm, hash_lm_dict, get_gm_stats = self.get_globalmodel(ciph_lm_dict, sign_lm_dict, iterkey=iterkey)
            
            # # [all DI] Verify global model locally (only simulating DIs[0])
            gm_soln, ver_sign_gm, ver_sign_gm_stat = self.ver_sign_globalmodel(gm, sign_gm, hash_lm_dict)

            # # Correctness Test
            # # test: Correctness of global model (part) aggregation
            print('* Test correctness of global model (part) aggregation')
            agg_all_lm = self.aggregate(solns)
            
            print('  agg_all_lm:', [round(float(x),4) for x in agg_all_lm[:10]])
            agg_solns = []
            for i in range(len(selected_clients)):
                if selected_clients[i].cid not in excluded_clients_ids:
                    agg_solns.append(solns[i])
            agg_part_lm = self.aggregate(agg_solns)
            print('  correct_gm:', [round(float(x),4) for x in agg_part_lm[:10]])
            print('  compute_gm:', [round(float(x),4) for x in gm_soln[:10]])
            dis1 = torch.dist(agg_part_lm, agg_all_lm, p=2)
            print('  distance between correct_gm and agg_all_lm:', float(dis1))
            dis2 = torch.dist(agg_part_lm, gm_soln, p=2)
            print('  distance between correct_gm and compute_gm:', float(dis2))

            # Track communication cost
            stats_dict = {  '[DI] enc&sign of lm&ik ': local_stats, 
                            '[DI] decshare of ciph_ik': decshare_stats, 
                            '[DI] ver sign_gm': ver_sign_gm_stat, 
                            '[SV] ver signs of lm&iterkey': ver_sign_local_stats,
                            '[SV] agg ciph_ik': agg_ciph_s_stats,
                            '[SV] get iterkey': get_iterkey_stats,
                            '[SV] dec&sign gm': get_gm_stats,
                            'MSE': str(float(dis2))}
            #stats_list = [stats,local_enc_stats,eva_stats,decshare_stats,get_gm_stats]        
            self.log_stats(round_i, selected_clients, stats_dict)
            file_path = mkdir(self.stat_dir)+'sa_stats.xlsx'
            self.log_stats_sheet(round_i, stats_dict, file_path=file_path)

            print('\n>>> Round: {} / MSE: {}'.format(round_i, float(dis2)) )
            print('======================================================================================================\n')



