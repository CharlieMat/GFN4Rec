import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from model.simulator import *
from env.KRUserEnvironment_ListRec import KRUserEnvironment_ListRec


class MLUserEnvironment_ListRec(KRUserEnvironment_ListRec):
    '''
    KuaiRand simulated environment on GPU machines
    Components:
    - multi-behavior user response model: 
        - (user history, user profile) --> user_state
        - (user_state, item) --> feedbacks (e.g. click, long_view, like, ...)
    - user leave model:
        - user temper reduces to <1 and leave
        - user temper drops gradually through time and further drops when the user is unsatisfactory about a recommendation
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - uirm_log_path
        - initial_temper
        - slate_size
        - max_step_per_episode
        - episode_batch_size
        - item_correlation
        - new_reader_class
        - env_val_holdout
        - env_test_holdout
        '''
        parser = KRUserEnvironment_ListRec.parse_model_args(parser)
        return parser
    
    def __init__(self, args):
        super().__init__(args)
        self.response_weights = [1 for f in self.response_types]
        
        
        