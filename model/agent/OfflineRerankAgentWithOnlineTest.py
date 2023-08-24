import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import utils
from model.agent.reward_func import *
from model.agent.OfflineAgentWithOnlineTest import OfflineAgentWithOnlineTest

class OfflineRerankAgentWithOnlineTest(OfflineAgentWithOnlineTest):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseOnlineAgent:
            - n_iter
            - train_every_n_step
            - start_train_at_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - save_episode
            - save_path
            - batch_size
            - actor_lr
            - actor_decay
        '''
        parser = OfflineAgentWithOnlineTest.parse_model_args(parser)
        parser.add_argument('--learn_initial_during_rerank', action='store_true', 
                            help='learning initial ranker when learning reranker')
        return parser
    
    def __init__(self, *input_args):
        args, actor, env, buffer = input_args
        # env, actor, buffer, device
        # # n_iter, train_every_n_step, start_train_at_step, 
        # check_episode, test_episode, save_episode, save_path, 
        # episode_batch_size, batch_size, actor_lr, actor_decay, 
        # reward_func, single_response, explore_rate
        # response_types, W, W_sum, NDCG_discount, rank_discount
        args, actor, env, buffer = input_args
        self.learn_initial_during_rerank = args.learn_initial_during_rerank
        super().__init__(args, actor, env, buffer)
    
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        # training records
        self.training_history = {}
        self.eval_history = {'avg_reward': [], 'max_reward': [], 'reward_variance': [], 
                             'coverage': [], 'intra_slate_diversity': [], 
                             'NDCG': [], 'MRR': []}
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        K = self.env.action_dim
        self.eval_history.update({f'NDCG_{t}': [] for t in range(K)})
        self.eval_history.update({f'MRR_{t}': [] for t in range(K)})
        self.initialize_training_history()
    
        # random explore before training
        initial_epsilon = 1.0
        self.env.reader.set_phase('train')
        self.offline_iter = iter(DataLoader(self.env.reader, batch_size = self.batch_size, shuffle = True, 
                                            pin_memory = True, num_workers = 8))
        self.actor.train_initial = True
        self.actor.train_rerank = False
        for i in tqdm(range(self.start_train_at_step)):
            # offline training and online evaluation
            self.step_train()
        self.actor.train_initial = self.learn_initial_during_rerank
        self.actor.train_rerank = True
        return None
    
    