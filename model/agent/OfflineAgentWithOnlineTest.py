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
from model.agent.BaseOnlineAgent import BaseOnlineAgent

class OfflineAgentWithOnlineTest(BaseOnlineAgent):
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
        parser = BaseOnlineAgent.parse_model_args(parser)
        return parser
    
    def __init__(self, *input_args):
        args, actor, env, buffer = input_args
        # env, actor, buffer, device
        # # n_iter, train_every_n_step, start_train_at_step, 
        # check_episode, test_episode, save_episode, save_path, 
        # episode_batch_size, batch_size, actor_lr, actor_decay, 
        # reward_func, single_response, explore_rate
        super().__init__(args, actor, env, buffer)
        # [response_name]
        self.response_types = self.env.immediate_response_model.feedback_types
        self.W = self.immediate_response_weight.detach().cpu().numpy()
        self.W_sum = sum(self.W)
        self.NDCG_discount = (-torch.arange(self.batch_size)).exp().to(self.device) + 1e-6
        self.rank_discount = (torch.arange(self.batch_size)+1).pow(-1.0).to(self.device)
        
    def train(self):
        # load model parameters if continue training
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        
        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            self.epsilon = self.exploration_scheduler.value(i)
            # offline training and online evaluation
            self.step_train()
            # log report
            if i % self.check_episode == 0 and i >= self.check_episode:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report()
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime
            # save model and training info
            if i % self.save_episode == 0:
                self.save()
        # offline evaluation
        self.action_after_train()
    
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
        self.offline_iter = iter(DataLoader(self.env.reader, batch_size = self.batch_size, shuffle = True, 
                                            pin_memory = True, num_workers = 8))
        self.env.reader.set_phase('train')
        return None
    
    
    def action_after_train(self):
        self.env.stop()
        # offline test
        self.test('test')
        
        
        
    ###############################
    #   Requires implementation   #
    ###############################
    
    
    def step_train(self):
        '''
        @process:
        - offline training:
            - batch_sample: {
                'user_id': (B,)
                'item_id': (B,slate_size)
                'uf_{feature}': (B,F_dim(feature)), user features
                'if_{feature}': (B,slate_size,F_dim(feature)), item features
                '{response}': (B,slate_size)
                'history': (B,max_H)
                'history_length': (B,)
                'history_if_{feature}': (B, max_H * F_dim(feature))
                'history_{response}': (B, max_H)
            }
            - policy.get_forward(): observation, candidates --> policy_output
            - policy.get_loss(): observation, candidates, policy_output, user_response --> loss
            - optimizer.zero_grad(); loss.backward(); optimizer.step()
            - update training history
        - 
        '''
        try:
            batch_sample = next(self.offline_iter)
        except:
            self.offline_iter = iter(DataLoader(self.env.reader, batch_size = self.batch_size, shuffle = True, 
                                                pin_memory = True, num_workers = 8))
            batch_sample = next(self.offline_iter)
            
        B = batch_sample['user_id'].shape[0]
        self.env.episode_batch_size = B
        sample_observation = self.env.get_observation_from_batch(batch_sample)
        
        # offline training
        # (B, slate_size)
        target_action = batch_sample['item_id'] - 1
        # (B, slate_size, response_dim)
        target_response = torch.cat([batch_sample[resp].view(B,self.env.action_dim,1) \
                                     for i,resp in enumerate(self.response_types)], dim = 2)
        user_feedback = {'immediate_response': target_response,
                         'immediate_response_weight': self.immediate_response_weight}
        user_feedback['reward'] = self.reward_func(user_feedback).detach()

        # forward pass
        sample_observation['batch_size'] = B
        candidate_info = self.env.get_candidate_info(None)
        input_dict = {'observation': sample_observation, 
                      'candidates': candidate_info, 
                      'action_dim': self.env.action_dim,
                      'action': target_action, 
                      'response': target_response,
                      'reward': user_feedback['reward'],
                      'epsilon': 0, 'do_explore': False, 'is_train': True}
        policy_output = self.actor(input_dict)

        # loss
        policy_output['action'] = target_action
        policy_output.update(user_feedback)
        loss_dict = self.actor.get_loss(input_dict, policy_output)
        actor_loss = loss_dict['loss']
        
        # optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for k in loss_dict:
            try:
                self.training_history[k].append(loss_dict[k].item())
            except:
                self.training_history[k].append(loss_dict[k])

        # online test
        self.online_one_step_eval(sample_observation, candidate_info)
    
    def online_one_step_eval(self, observation, candidate_info):
        '''
        Run one step of user-env interaction
        @input:
        - observation: same as self.env.reset@output - observation
        - candidate_info: same as self.env.get_candidate_info@output
        @process:
        - policy.explore_action(): observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        '''
        # online evaluation
        with torch.no_grad():
            self.env.current_observation = observation
            
            # sample action
            input_dict = {'observation': observation, 
                          'candidates': candidate_info, 
                          'action_dim': self.env.action_dim,
                          'action': None, 'response': None,
                          'epsilon': 0, 'do_explore': False, 'is_train': False}
            policy_output = self.actor(input_dict)
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['action']}   

            user_feedback = self.env.get_response(action_dict)
            # calculate reward
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()
            user_feedback['reward'] = R
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['max_reward'].append(R.max().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            self.eval_history['coverage'].append(user_feedback['coverage'])
            self.eval_history['intra_slate_diversity'].append(user_feedback['ILD'])
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())
                
            # offline metrics
            # (B, slate_size)
            target_action = policy_output['action']
            # (B, slate_size, response_dim)
            target_response = user_feedback['immediate_response']
            metric_out = self.get_offline_metrics(observation, candidate_info, 
                                                  target_action, target_response)
            for k,v in metric_out.items():
                if k in self.eval_history:
                    self.eval_history[k].append(v)

    def get_offline_metrics(self, observation, candidate_info, target_action, target_response):
        B = target_action.shape[0]
        K = self.env.action_dim
        # (B, slate_size)
        point_reward = torch.sum(target_response * self.immediate_response_weight.view(1,1,-1), dim = 2).view(B,K)
        # forward
        input_dict = {'observation': observation, 
                      'candidates': candidate_info, 
                      'action_dim': self.env.action_dim,
                      'action': target_action, 'response': None,
                      'epsilon': 0, 'do_explore': False, 'is_train': False}
        policy_output = self.actor(input_dict)
        # (B, slate_size)
        P = policy_output['prob'].view(B,K).detach()

        metric_dict = {'NDCG': 0, 'MRR': 0}
        metric_dict.update({f"NDCG_{t}": 0 for t in range(K)})
        metric_dict.update({f"MRR_{t}": 0 for t in range(K)})
        
        RAUC = 0
        NDCG = 0
        MRR = 0
        for t in range(K):
            P_t, P_indices = torch.sort(P[:,t], descending = True)

            # NDCG
            ranking_metrics = self.get_ranking_metrics(P[:,t], point_reward[:,t], self.NDCG_discount)
            metric_dict[f'NDCG_{t}'] = ranking_metrics['NDCG']
            metric_dict[f'MRR_{t}'] = ranking_metrics['MRR']
            NDCG += ranking_metrics['NDCG']
            MRR += ranking_metrics['MRR']
#         metric_dict['RAUC'] = RAUC / K
        metric_dict['NDCG'] = NDCG / K
        metric_dict['MRR'] = MRR / K
        return metric_dict
    
    def get_ranking_metrics(self, sorted_P, relevance, NDCG_discount):
        # scalar
        DCG = torch.sum(relevance * NDCG_discount[:len(sorted_P)])
        # (B, )
        sorted_reward, _ = torch.sort(relevance, descending = True)
        # scalar
        IDCG = torch.sum(sorted_reward * NDCG_discount[:len(sorted_P)])
        # scalar
        NDCG = (DCG / IDCG).item()
        # scalar
        MRR = torch.mean(relevance * self.rank_discount[:len(sorted_P)]).item()
        return {'NDCG': NDCG, 'MRR': MRR}
    
    def test(self, *episode_args):
        '''
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - policy.explore_action(): observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        '''
        phase = episode_args[0]
        assert phase == 'val' or phase == 'test'
        self.env.reader.set_phase(phase)
        self.offline_test_iter = DataLoader(self.env.reader, batch_size = self.batch_size, shuffle = True, 
                                       pin_memory = True, num_workers = 8)
        
        K = self.env.action_dim # slate size
        test_report = {"NDCG": [], "MRR": []}
        test_report.update({f"NDCG_{t}": [] for t in range(K)})
        test_report.update({f"MRR_{t}": [] for t in range(K)})
        
        with torch.no_grad():
            # entire item pool as shared candidate info
            candidate_info = self.env.get_candidate_info(None)
            
            for i,test_batch in tqdm(enumerate(self.offline_test_iter)):
                B = test_batch['user_id'].shape[0] # batch size
                observation = self.env.get_observation_from_batch(test_batch)
                # (B, slate_size)
                target_action = test_batch['item_id'].view(B,K) - 1
                # (B, slate_size, response_dim)
                target_response = torch.cat([test_batch[resp].view(B,K,1) \
                                             for i,resp in enumerate(self.response_types)], dim = 2)
                
                metric_out = self.get_offline_metrics(observation, candidate_info, target_action, target_response)
                for k,v in metric_out.items():
                    if k in test_report:
                        test_report[k].append(v)
                
        test_report = {k: np.mean(v) for k,v in test_report.items()}
        log_str = f"{test_report}"
        print(f"Offline test_result:\n{log_str}")
        with open(self.save_path + "_offline_test.report", 'w') as outfile:
            outfile.write(log_str)
        return None
    

    def save(self):
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
        
        