from mkTPFL.ckks.ckks_parameters import * # for test
from mkTPFL.trainers.base import BaseMKTPFLTrainer

import time
import torch
from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm
import os
import sys

from mkTPFL.utils.class_transformer import *


class BasicSATrainer(BaseMKTPFLTrainer):
    def __init__(self, options, dataset, verify_data=None):
        dataset_name = options['dataset']

        pfl_options = defaultdict(lambda:False)

        # (BASICSATrainer do not support dropouts)
        pfl_options['dropout_ratio'] = 0 

        # (SATrainer do not evaluate local models)
        pfl_options['eval_client'] = False 
        # accuracy of input vectors
        pfl_options['soln_scaling'] = 10**6
        # drop-out simulation
        if 'dropout_ratio' not in options:
            options['dropout_ratio'] = 0.5

        options.update(pfl_options)
        super(BasicSATrainer, self).__init__(options, dataset)
        

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




    def local_partencrypt_soln(self, selected_clients, solns, eval_mem_flag=False, save_flag=False, round_num='tmp'):
        ''' each DI: 
            0. rescale: lm = [round(float(v)*self.soln_scaling) for v in soln]
            1. ciph_lm = partEncrypt(lm)

            output: ciph_lm_dict, sign_clm_dict, stats
                - ciph_lm_dict = {cid:ciph_lm}
                    ciph_lm = ciph_lm_file if save_flag else ciph_lm
                - stats[cid] includes 'bytes_lm', 'time_enc_lm', 'bytes_enc_lm'
        '''
        # stats
        stats = defaultdict(dict)
        eval_mem_flag = eval_mem_flag if eval_mem_flag else self.eval_mem_flag

        print('(DIs) partEncrypt local_model:')
        ciph_lm_dict = {} # ciphertexts of local models, ciph_lm_dict[cid] = local model ciphertxt

        log_flag = False # only log one client
        for ci in tqdm(range(len(selected_clients))):
            cid, soln = selected_clients[ci].cid, solns[ci][1]
            di, lm = self.dis[cid], [round(float(v)*self.soln_scaling) for v in soln]
            stats[cid].update({'bytes_lm':sys.getsizeof(lm)})

            # encrypt(lm)
            begin_time = time.time()
            # ciph_lm = di.partEncrypt(lm, thread_num=1)
            ciph_lm = di.partEncrypt(lm)
            ciph_lm_dict[cid] = ciph_lm
            end_time = time.time()
            if not log_flag:
                stats[cid].update({'time_enc_lm':round(end_time-begin_time, 4)})
            log_flag = True

            if save_flag: # save to file
                ciph_lm_data = DataCiphModel(ciph_lm, print_flag=True)
                di_dir = mkdir(self.local_data_dir+'DI_'+str(di.di_index)+'/')
                di_file_path = di_dir +  'round_' + str(round_num) + 'ciph_lm_data.pkl'
                self.trainer_save_data(ciph_lm_data, file_path=di_file_path, print_flag=True)
                del ciph_lm_dict[cid]
                ciph_lm_dict[cid] = di_file_path

        # tqdm.write('\033[F', end='\r') # remove the prompt in tqdm

        if eval_mem_flag: # statistics memory size
            cid = selected_clients[0].cid # sample client
            di = self.dis[cid] 
            ciph_lm_data = DataCiphModel(ciph_lm_dict[cid], print_flag=True)
            di_dir = mkdir(self.local_data_dir+'DI/')
            di_file_path = di_dir + 'ciph_lm_data.pkl'
            self.trainer_save_data(ciph_lm_data, file_path=di_file_path, print_flag=True)
            stats[cid].update({'bytes_enc_lm': os.path.getsize(di_file_path)})

        return ciph_lm_dict, stats


    def aggregation_partciph_lm(self, partciph_lm_dict, eval_mem_flag=False):
        ''' SV: 
                1. ciphAggregate(partciph_lm_dict): aggregates the ciphertexts of iterkey
                
                output: ciph_gm, stats
                - ciph_gm = ciphAggregate(partciph_lm_dict)
                - stats includes 'time_agg_ciph_lm'
        '''
        sv = self.svs[0]
        eval_mem_flag = eval_mem_flag if eval_mem_flag else self.eval_mem_flag
        print('(SV) Aggregate ciphertexts of local models.')
        
        # aggregate ciphertexts
        if self.clients_weights:
            partciph_lm_list, weights = [],[]
            for cid, partciph_lm in partciph_lm_dict.items():
                partciph_lm_list.append(partciph_lm)
                weights.append(self.clients_weights[cid])
        else:
            partciph_lm_list = list(partciph_lm_dict.values())
            weights = None

        begin_time = time.time()
        ciph_gm = sv.ciphAggregate(partciph_lm_list, weights=weights)
        time_agg_ciph_lm = round(time.time()-begin_time, 4)

        stats = { 'time_agg_ciph_lm':time_agg_ciph_lm }
        
        return ciph_gm, stats


    def get_globalmodel(self, ciph_gm, decshare_lm_dict, eval_mem_flag=False):
        ''' SV: 
                get the aggregation decrypt key 
        '''
        print('(SV) Get global model: ')
        eval_mem_flag = eval_mem_flag if eval_mem_flag else self.eval_mem_flag

        begin_time = time.time()
        decshare_list = list(decshare_lm_dict.values())

        gm = [] 
        dvecs = self.svs[0].decshareMerge(ciph_gm, decshare_list, plain_flag=False, msg_flag=False)
        if not isinstance(dvecs, list):
            dvecs = [dvecs]
        for dvec in dvecs:
            dvec_size = min( dvec.size(), self.model_len-len(gm) )
            gm += [ round(float(str(dvec[j].real()))) for j in range(dvec_size) ]
        time_get_gm = round(time.time()-begin_time, 4)
        print('\t done! using {} seconds'.format(time_get_gm))

        stats = { 'time_get_gm':time_get_gm, 'bytes_gm_soln': sys.getsizeof(gm) }

        if self.clients_weights:
            weights = [self.clients_weights[cid] for cid in decshare_lm_dict.keys()]
            sum_weights = sum(weights)
        else:
            sum_weights = len(decshare_lm_dict)
        gm_soln = (torch.tensor(gm)/sum_weights)/self.soln_scaling

        return gm, gm_soln, stats


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        for round_i in range(self.num_round):
        # for round_i in range(1):
            print('======================================================================================================')
            print('ROUND [{}]'.format(round_i))
            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=self.seed+round_i)

            # Generate the aggregation public key
            self.set_agg_pk(selected_clients)

            # Input locally
            if self.dataset_name not in ['randomvector','rv']: # real model
                self.latest_model = self.worker.get_flat_model_params().detach()
                solns, stats = self.local_train(round_i, selected_clients)
            else: # random_vector
                self.worker = None
                self.latest_model = np.array([ round(x, 4) for x in np.random.rand(self.input_length) ])
                solns = self.local_input_solns(selected_clients, seed=self.seed+round_i)
                
                
            # # [selected DI] Encrypt locally
            # partEncrypt(lm)
            partciph_lm_dict, local_lm_stats = self.local_partencrypt_soln(selected_clients, solns, save_flag=False, round_num=round_i)

            # # [SV] Aggregation iterkey ciphertexts: ciph_s_sum = aggregation(partciph_s) for clients who passed signuature verification & local model evaluation
            ciph_gm, agg_ciph_lm_stats = self.aggregation_partciph_lm(partciph_lm_dict)

            # # [online DI] Decrypt shares locally
            # [selected DI] calculate partdecshare of ciph_iterkey locally
            decshare_lm_dict, decshare_stats = self.local_decshare(ciph_gm, recovered_clients=selected_clients)
            
            # # [SV] Get global model
            gm, gm_soln, get_gm_stats = self.get_globalmodel(ciph_gm, decshare_lm_dict)

            # # Correctness Test
            # # test: Correctness of global model (part) aggregation
            print('* Test correctness of global model (part) aggregation')
            agg_solns = []
            for i in range(len(selected_clients)):
                agg_solns.append(solns[i])
            agg_part_lm = self.aggregate(agg_solns)
            print('  correct_gm:', [round(float(x),4) for x in agg_part_lm[:10]])
            print('  compute_gm:', [round(float(x),4) for x in gm_soln[:10]])
            dis = torch.dist(agg_part_lm, gm_soln, p=2)
            print('  distance between correct_gm and compute_gm:', float(dis))

            # Track communication cost
            stats_dict = {  '[DI] enc of lm ': local_lm_stats, 
                            '[DI] decshare of ciph_lm': decshare_stats, 
                            '[SV] agg ciph_lm': agg_ciph_lm_stats,
                            '[SV] get gm': get_gm_stats,
                            'MSE': str(float(dis))}
            #stats_list = [stats,local_enc_stats,eva_stats,decshare_stats,get_gm_stats]        
            self.log_stats(round_i, selected_clients, stats_dict)
            file_path = mkdir(self.stat_dir)+'basic_sa_stats.xlsx'
            self.log_stats_sheet(round_i, stats_dict, file_path=file_path)

            print('\n>>> Round: {} / MSE: {}'.format(round_i, float(dis)) )
            print('======================================================================================================\n')



