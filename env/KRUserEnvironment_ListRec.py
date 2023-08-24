import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from env.KRUserEnvironment_FiniteImmediate import KRUserEnvironment_FiniteImmediate


class KRUserEnvironment_ListRec(KRUserEnvironment_FiniteImmediate):
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
        parser = KRUserEnvironment_FiniteImmediate.parse_model_args(parser)
        return parser
    
    def __init__(self, args):
        '''
        self.device
        self.initial_temper
        self.slate_size
        self.max_step_per_episode
        self.episode_batch_size
        self.rho
        self.immediate_response_stats
        self.immediate_response_model
        self.max_hist_len
        self.response_types
        self.response_dim
        self.response_weights
        self.reader
        self.candidate_iids
        self.candidate_item_meta
        self.n_candidate
        self.candidate_item_encoding
        self.gt_state_dim
        self.action_dim
        self.observation_space
        self.action_space
        '''
        super().__init__(args)
        
        
    def reset(self):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': the episode running batch size, 
                    'empty_history': True if start from empty history, default = False
                    'initial_history': start with initial history, empty_history must be False}
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        @update:
        - self.current_observation: same as @output - observation
        - self.temper
        '''
        BS = self.episode_batch_size
        self.iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True, 
                                          pin_memory = True, num_workers = 8))
        initial_sample = next(self.iter)
        self.current_observation = self.get_observation_from_batch(initial_sample)
        self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.user_step_count = 0
        return deepcopy(self.current_observation)
    
    
    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, W_dim)}
        '''
        # (B, slate_size)
        action = step_dict['action']
        
        # user interaction
        with torch.no_grad():
            response_out = self.get_response(step_dict)
            # (B, slate_size, n_feedback)
            response = response_out['immediate_response']

            # get leave signal
            # (B,), 0-1 vector
            done_mask = self.get_leave_signal(response)
            
            # update observation
            update_info = self.update_observation(action, response, done_mask)
            self.user_step_count += 1
            
            for i,f in enumerate(self.response_types):
                # (B, )
                R = response.mean(1)[:,i].detach()

            if done_mask.sum() == len(done_mask):
                new_iter_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != len(done_mask):
                        new_sample_flag = True
                except:
                    new_sample_flag = True
                if new_sample_flag:
                    self.iter = iter(DataLoader(self.reader, batch_size = done_mask.shape[0], shuffle = True, 
                                                pin_memory = True, num_workers = 8))
                    sample_info = next(self.iter)
                new_observation = self.get_observation_from_batch(sample_info)
                self.current_observation = new_observation
                self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
                self.user_step_count = 0
            elif done_mask.sum() > 0:
                print(done_mask)
                print("User leave not synchronized")
                raise NotImplemented
        user_feedback = {'immediate_response': response, 
                         'done': done_mask, 
                         'coverage': response_out['coverage'], 
                         'ILD': response_out['ILD']}
        return deepcopy(self.current_observation), user_feedback, update_info['updated_observation']

    def get_env_report(self, window = 50):
        report = super().get_env_report(window)
        return report
        
        
        
        
        