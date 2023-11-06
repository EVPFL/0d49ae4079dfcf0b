"""A module to keep track of parameters for the CKKS scheme."""
from mkTPFL.ckks.ckks_parameters import *
from mkTPFL.mklhs.mklhs_parameters import *
from mkTPFL.utils.class_transformer import *

from mkTPFL.roles.dataisland import DataIsland
from mkTPFL.roles.flserver import FLServer

from HEAAN import SerializationUtils

from fedAvg.utils.worker_utils import mkdir

import numpy as np
import os



def data2EvalDatasets(eval_data):
    return eval_data


#### ------------------------------------------------------------
#### ------------  ROLE & ROLE_INFO TRANSFER METHODS ------------
#### ------------------------------------------------------------ 
#### ------------------ TRANSFER DATA TO ROLE -------------------
def data2Role(data, role_type):
    role = None
    if role_type == 'DI':
        role = {}
        for (cid,di_data) in data.items():
            di = DataIsland(di_data['di_index'])    
            if 'key_files_flag' in di_data and di_data['key_files_flag']:
                di.secret_key_file = di_data['secret_key_file']
                di.public_key_file =  di_data['public_key_file']
                di.enc_secret_key_file = di_data['enc_secret_key_file']
                di.enc_public_key_file = di_data['enc_public_key_file']
                di.enc_relin_key_file = di_data['enc_relin_key_file']
                di.enc_rot_keys_file = di_data['enc_rot_keys_file']
                di.sign_secret_key_file = di_data['sign_secret_key_file']
                di.sign_public_key_file = di_data['sign_public_key_file']
            else:
                di.__secret_key = data2Obj(di_data['secret_key'], 'SecretKey')
                di.__public_key = data2Obj(di_data['public_key'], 'Key')
                di.__enc_secret_key = data2Obj(di_data['enc_secret_key'], 'SecretKey')
                di.__enc_public_key = data2Obj(di_data['enc_public_key'], 'Key')
                di.__enc_relin_key = data2Obj(di_data['enc_relin_key'], 'Key')
                di.__enc_rot_keys = [ data2Obj(rk, 'Key') for rk in di_data['enc_rot_keys'] ]
                #di.__enc_rot_key = data2Obj(di_data['enc_rot_key'], 'Key')
                di.__sign_secret_key = data2Obj(di_data['sign_secret_key'], 'LHSSKey')
                di.__sign_public_key = data2Obj(di_data['sign_public_key'], 'LHSPKey')
            if 'di_secret_key_shares_file' in di_data:
                di.di_secret_key_shares_file = di_data['di_secret_key_shares_file']
            role[cid] = di
    if role_type == 'SV':
        role = []
        for sv_data in data:
            sv = FLServer(sv_data['sv_index'])
            if 'key_files_flag' in sv_data and sv_data['key_files_flag']:
                sv.secret_key_file = sv_data['secret_key_file'] if 'secret_key_file' in sv_data else None
                sv.public_key_file = sv_data['public_key_file']if 'public_key_file' in sv_data else None
                sv.di_secret_key_shares_file = sv_data['di_secret_key_shares_file'] if 'di_secret_key_shares_file' in sv_data else None
            else:
                sv.__secret_key = data2Obj(sv_data['secret_key'], 'SecretKey') if 'secret_key' in sv_data else None
                sv.__public_key = data2Obj(sv_data['public_key'], 'Key') if 'public_key' in sv_data else None
                sv.__di_secret_key_shares = [ data2Obj(sk, 'SecretKey') for sk in sv_data['di_secret_key_shares'] ] if 'di_secret_key_shares' in sv_data else None
            role.append(sv)
    if role_type == 'CKKSParams':
        logq, logp, logn = data['logq'], data['logp'], data['logn']
        ax, rax = ZZ(list(data['ax'])), Uint64_t(data['rax'])
        role = CKKSParameters(logq, logp, logn, ax=ax, rax=rax)
    if role_type == 'MKLHSParams':
        # if not mklhs_init_res:
        #     mklhs_init_res = init_mklhs()
        global mklhs_scheme
        if not mklhs_scheme:
            mklhs_scheme = MKLHS_init()
        params = LHSParams()
        # params.setN(data['n_str'])
        params.setDataset(data['dataset'])
        params.setLabel(data['label'])
        role = MKLHSParameters(params=params)
    return role
#### ------------------------------------------------------------ 

#### ------------------ TRANSFER ROLE TO DATA -------------------
def role2Data(role, role_type, key_files_flag=True):
    role_data = {}
    if role_type == 'CKKSParams':
        role_data['logq'] = role.logq
        role_data['logp'] = role.logp
        role_data['logn'] = role.logn
        role_data['rax'] = np.array([ int(role.rax[i]) for i in range(Nnprimes) ])
        role_data['ax'] = np.array([ int(role.ax[i]) for i in range(N) ])

    if role_type == 'MKLHSParams':
        # role_data['n_str'] = role.n_str
        role_data['dataset'] = role.dataset
        role_data['label'] = role.label

    if role_type == 'DI':
        role_data['key_files_flag'] = key_files_flag
        role_data['di_index'] = role.di_index
        if not key_files_flag:
            role_data['secret_key'] = DataSecretKey(role.secret_key)
            role_data['public_key'] = DataKey(role.public_key)
            role_data['enc_secret_key'] = DataSecretKey(role.enc_secret_key)
            role_data['enc_public_key'] = DataKey(role.enc_public_key)
            role_data['enc_relin_key'] = DataKey(role.enc_relin_key) if role.enc_relin_key else None
            role_data['enc_rot_keys'] = [ DataKey(rk) for rk in role.enc_rot_keys ] if role.enc_rot_keys else None
            role_data['sign_secret_key'] = DataLHSSKey(role.sign_secret_key)
            role_data['sign_public_key'] = DataLHSPKey(role.sign_public_key)
        else:
            role_data['secret_key_file'] = role.secret_key_file
            role_data['public_key_file'] = role.public_key_file
            role_data['enc_secret_key_file'] = role.enc_secret_key_file
            role_data['enc_public_key_file'] = role.enc_public_key_file
            role_data['enc_relin_key_file'] = role.enc_relin_key_file
            role_data['enc_rot_keys_file'] = role.enc_rot_keys_file
            role_data['sign_secret_key_file'] = role.sign_secret_key_file
            role_data['sign_public_key_file'] = role.sign_public_key_file
        
        if role.sv_num == 1:
            role_data['di_secret_key_shares_file'] = role.di_secret_key_shares_file
        
    if role_type == 'SV':
        role_data['sv_index'] = role.sv_index
        if not key_files_flag:
            role_data['secret_key'] = DataSecretKey(role.secret_key) if role.secret_key else None
            role_data['public_key'] = DataKey(role.public_key)  if role.public_key else None
        else:
            role_data['secret_key_file'] = role.secret_key_file
            role_data['public_key_file'] = role.public_key_file

        if role.sv_num > 1:
            if not key_files_flag and role.sv_num <= role.sv_threshold:
                role_data['di_secret_key_shares'] = [ DataSecretKey(sk) for sk in role.di_secret_key_shares_file ]
            else:
                role_data['di_secret_key_shares_file'] = role.di_secret_key_shares_file

    return role_data
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------ 
#### ------------------------------------------------------------

#### ------------------------------------------------------------ 
#### -------------------- KEYS & KEYS FILE --------------------- 
#### ------------------------------------------------------------ 
def saveDIKeys(di, dir_path, print_flag=False, reset_flag=True):
    print(' - saving DI['+str(di.di_index)+'] keys into file ...')
    di_dir_path = mkdir(dir_path + 'DI_'+str(di.di_index)+'/' )
    keys_data = {}
    keys_data['secret_key'] = DataSecretKey(di.secret_key)
    keys_data['public_key'] = DataKey(di.public_key)
    keys_data['enc_secret_key'] = DataSecretKey(di.enc_secret_key)
    keys_data['enc_public_key'] = DataKey(di.enc_public_key)
    keys_data['enc_relin_key'] = DataKey(di.enc_relin_key) if di.enc_relin_key else None
    keys_data['enc_rot_keys'] = [ DataKey(rk) for rk in di.enc_rot_keys ] if di.enc_rot_keys else None
    keys_data['sign_secret_key'] = DataLHSSKey(di.sign_secret_key)
    keys_data['sign_public_key'] = DataLHSPKey(di.sign_public_key)
    if di.sv_num > 1 and di.sv_num <= di.sv_threshold:
        keys_data['secret_key_shares'] = [ DataSecretKey(sk) for sk in di.secret_key_shares ]

    keys2Files = {}
    for kname,kdata in keys_data.items():
        file_path = di_dir_path + kname +'_data.pkl'
        saveData(kdata, file_path=file_path, print_flag=print_flag)
        keys2Files[kname] = file_path

    if reset_flag:
        di.resetKeys2Files(keys2Files)


def setDIKeyFiles(di, dir_path, print_flag=False):
    di_dir_path = dir_path + 'DI_'+str(di.di_index)+'/'
    if not os.path.exists(di_dir_path):
        print(' >>> no DI Keys dir [{}] , generating new keys ...'.format(di_dir_path))
        return False

    print(' >>> set DI[{}] key files from dir [{}]'.format(di.di_index, di_dir_path) )
    keynames = ['secret_key', 
                'public_key',
                'enc_secret_key',
                'enc_public_key',
                'enc_relin_key',
                'enc_rot_keys',
                'sign_secret_key',
                'sign_public_key',
                'secret_key_shares']
    keys2Files = {}
    for kname in keynames:
        file_path = di_dir_path + kname + '_data.pkl'
        if os.path.exists(file_path):
            keys2Files[kname] = file_path
            # print('   - {}: {}'.format(kname, file_path))
    di.resetKeys2Files(keys2Files)
    return True


def saveSVKeys(sv, dir_path, print_flag=False, reset_flag=True):
    print(' - saving SV['+str(sv.sv_index)+'] keys into file ...')
    sv_dir_path = mkdir(dir_path + 'SV_'+str(sv.sv_index)+'/' )
    keys_data = {}
    keys_data['secret_key'] = DataSecretKey(sv.secret_key) if sv.secret_key else None
    keys_data['public_key'] = DataKey(sv.public_key) if sv.public_key else None
    if sv.sv_num > 1 and sv.sv_num <= sv.sv_threshold:
        keys_data['di_secret_key_shares'] = [ DataSecretKey(sk) for sk in sv.di_secret_key_shares ]

    keys2Files = {}
    for kname,kdata in keys_data.items():
        file_path = sv_dir_path + kname +'_data.pkl'
        saveData(kdata, file_path=file_path, print_flag=print_flag)
        keys2Files[kname] = file_path

    if reset_flag:
        sv.resetKeys2Files(keys2Files)

def setSVKeyFiles(sv, dir_path, print_flag=False):
    sv_dir_path = dir_path + 'SV_'+str(sv.sv_index)+'/'
    if not os.path.exists(sv_dir_path):
        print('>>> no SV Keys dir [{}] , generating new keys ...'.format(sv_dir_path))
        return False
    print(' >>> set SV[{}] key files from dir [{}]'.format(sv.sv_index, sv_dir_path) )
    keynames = ['secret_key', 
                'public_key',
                'di_secret_key_shares']
    keys2Files = {}
    for kname in keynames:
        file_path = sv_dir_path + kname + '_data.pkl'
        if os.path.exists(file_path):
            keys2Files[kname] = file_path
    sv.resetKeys2Files(keys2Files)
    return True
    
#### ------------------------------------------------------------
#### ------------------------------------------------------------


#### ------------------------------------------------------------
#### ----------------- UPDATE TRAINER FROM FILE -----------------
#### ------------------------------------------------------------
#### ------------------- UPDATE NEW TRAINER  --------------------
def build_trainer_from_file(trainer, file_path=None):
    """ update BaseMKTPFLTrainer from file
    """
    if not file_path or file_path == 'default':
        file_path = trainer.trainer_pfl_info_file+'_trainer_info_data.pkl'

    if not os.path.exists(file_path):
        print('>>> no setup file [{}].'.format(file_path))
        trainer.build_new_trainer()
        return

    # get trainer information from file
    trainer_info = trainer.trainer_get_data(file_path=file_path, print_flag=True)
    # trainer.ckks_params = data2Role(trainer_info['ckks_params'], 'CKKSParams')
    # trainer.mklhs_params = data2Role(trainer_info['mklhs_params'], 'MKLHSParams')
    # trainer.sv_num, trainer.di_num = trainer_info['sv_num'], trainer_info['di_num']
    # trainer.sv_threshold, trainer.di_threshold = trainer_info['sv_threshold'], trainer_info['di_threshold']
    trainer.svs, trainer.dis = data2Role(trainer_info['svs'], 'SV'), data2Role(trainer_info['dis'],'DI')
    
    # set attributes for svs and dis
    # get keys attributes from trainer information
    di_public_keys = [None] * trainer.di_num
    di_enc_public_keys = [None] * trainer.di_num
    di_enc_relin_keys = [None] * trainer.di_num
    di_enc_rot_keys = [None] * trainer.di_num
    di_sign_public_keys = [None] * trainer.di_num
    if not trainer.key_files_flag:
        for di in trainer.dis.values():
            di_public_keys[di.di_index] = di.public_key
            di_enc_public_keys[di.di_index] = di.enc_public_key
            di_enc_relin_keys[di.di_index] = di.enc_relin_key
            di_enc_rot_keys[di.di_index] = di.enc_rot_keys
            di_sign_public_keys[di.di_index] = di.sign_public_key
    else:
        for di in trainer.dis.values():
            di_public_keys[di.di_index] = di.public_key_file
            di_enc_public_keys[di.di_index] = di.enc_public_key_file
            di_enc_relin_keys[di.di_index] = di.enc_relin_key_file
            di_enc_rot_keys[di.di_index] = di.enc_rot_keys_file
            di_sign_public_keys[di.di_index] = di.sign_public_key_file

    # set DIs' general attributes
    for di in trainer.dis.values():
        di.di_num = trainer.di_num
        di.sv_num = trainer.sv_num
        di.di_threshold = trainer.di_threshold
        di.sv_threshold = trainer.sv_threshold
        di.ckks_params = trainer.ckks_params
        di.model_name = trainer.model_name
        if not trainer.key_files_flag:
            di.di_public_keys = di_public_keys
            di.di_sign_public_keys = di_sign_public_keys
        else:
            di.di_public_keys_file = di_public_keys
            di.di_sign_public_keys_file = di_sign_public_keys

    # set SVs' general attributes
    for sv in trainer.svs:
        sv.di_num = trainer.di_num
        sv.sv_num = trainer.sv_num
        sv.di_threshold = trainer.di_threshold
        sv.sv_threshold = trainer.sv_threshold
        sv.ckks_params = trainer.ckks_params
        sv.model_name = trainer.model_name

    # set sk_shares attributes
    if trainer.sv_num == 1 and trainer.di_threshold < trainer.di_num:
        for di in trainer.dis.values():
            di_dir_path = trainer.diskshares_dir + 'DI_'+str(di.di_index)+'/'
            di.di_secret_key_shares_dir = di_dir_path
    if trainer.sv_num > 1:
        for sv in trainer.svs:
            sv_dir_path = trainer.svkeys_dir_path + 'SV_'+str(sv.sv_index)+'/'
            sv.di_secret_key_shares_dir = sv_dir_path
    trainer.gen_di_sk_shares(update_flag=False)

    # set main server attributes
    sv = trainer.svs[0] 
    if not trainer.key_files_flag:
        sv.di_public_keys = di_public_keys
        sv.di_enc_public_keys = di_enc_public_keys
        sv.di_enc_relin_keys = di_enc_relin_keys
        sv.di_enc_rot_keys = di_enc_rot_keys
        sv.di_sign_public_keys = di_sign_public_keys
    else:
        sv.di_public_keys_file = di_public_keys
        sv.di_enc_public_keys_file = di_enc_public_keys
        sv.di_enc_relin_keys_file = di_enc_relin_keys
        sv.di_enc_rot_keys_file = di_enc_rot_keys
        sv.di_sign_public_keys_file = di_sign_public_keys







