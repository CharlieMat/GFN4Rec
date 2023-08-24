import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class PointwiseRanker(BaseOnlinePolicy):
    '''
    GFlowNet with Detailed Balance for listwise recommendation
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - ptranker_state2z_hidden_dims
        - ptranker_pos_offset
        - ptranker_neg_offset
        - from BaseOnlinePolicy:
            - from BackboneUserEncoder:
                - user_latent_dim
                - item_latent_dim
                - transformer_enc_dim
                - transformer_n_head
                - transformer_d_forward
                - transformer_n_layer
                - state_hidden_dims
                - dropout_rate
                - from BaseModel:
                    - model_path
                    - loss
                    - l2_coef
        '''
        parser = BaseOnlinePolicy.parse_model_args(parser) 
        parser.add_argument('--ptranker_state2z_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--ptranker_pos_offset', type=float, default=0.8, 
                            help='smooth offset of positive prob')
        parser.add_argument('--ptranker_neg_offset', type=float, default=0.1, 
                            help='smooth offset of negative prob')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args)
        self.ptranker_state2z_hidden_dims = args.ptranker_state2z_hidden_dims
        self.ptranker_pos_offset = args.ptranker_pos_offset
        self.ptranker_neg_offset = args.ptranker_neg_offset
        super().__init__(args, reader_stats, device)
        self.display_name = "PointwiseRanker"
        
    def to(self, device):
        new_self = super(PointwiseRanker, self).to(device)
        return new_self

    def _define_params(self, args):
        # userEncoder, enc_dim, state_dim, bce_loss
        super()._define_params(args)
        # p_forward
        self.state2z = DNN(self.state_dim, args.ptranker_state2z_hidden_dims, self.enc_dim, 
                           dropout_rate = args.dropout_rate, do_batch_norm = True)
        self.state2zNorm = nn.LayerNorm(self.enc_dim)

    
    def generate_action(self, user_state, feed_dict):
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        action_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'logP': (B, K), 
                     'logF': (B,),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        B = user_state.shape[0]
        # batch-wise candidates has shape (B,L), non-batch-wise candidates has shape (1,L)
        batch_wise = True
        if candidates['item_id'].shape[0] == 1:
            batch_wise = False
        # during training, candidates is always the full item set and has shape (1,L) where L=N
        if is_train:
            assert not batch_wise
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
            
        # (1,L,enc_dim) or (B,L,enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        # (B, enc_dim)
        Z = self.state2z(user_state)
        Z = self.state2zNorm(Z)
        # (B, L)
        score = torch.sum(Z.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
        # (B, L)
        prob = torch.softmax(score, dim = 1)
        
        if is_train or torch.is_tensor(action_slate):
            indices = action_slate
        else:
            if do_explore:
                # exploration: categorical sampling or uniform sampling
                if do_uniform:
                    indices = Categorical(torch.ones_like(prob)).sample((self.slate_size,)).transpose(0,1)
                else:
                    indices = Categorical(prob).sample((self.slate_size,)).transpose(0,1)
            else: 
                # greedy: topk selection
                _, indices = torch.topk(prob, k = self.slate_size, dim = 1)
            indices = indices.view(-1,self.slate_size).detach()
        selected_P = torch.gather(prob,1,indices)
        # slate action (B, K)
        slate_action = indices
                
        reg = self.get_regularization(self.state2z)

        out_dict = {'prob': selected_P, 
                    'action': slate_action, 
                    'reg': reg}
        
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Reward-based pointwise ranking loss
        * - Ylog(P) - (1-Y)log(1-P)
        * Y = sum(w[i] * r[i]) # the weighted sum of user responses
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'immediate_response_weight: (n_feedback, ),
            'reward': (B,)}
        @output
        - loss
        '''
        B = out_dict['prob'].shape[0]
        # (B,K)
        log_P = torch.log(out_dict['prob'] + self.ptranker_pos_offset)
        log_neg_P = torch.log(1 - out_dict['prob'] + self.ptranker_neg_offset)
        # (B,K,n_feedback)
        weighted_response = out_dict['immediate_response'].view(B,self.slate_size,-1) \
                                * out_dict['immediate_response_weight'].view(1,1,-1)
        # (B,K)
        Y = torch.mean(weighted_response, dim = 2)
        # (B,K)
        R_loss = - torch.mean(Y * log_P + (1-Y) * log_neg_P)
        # scalar
        loss = R_loss + self.l2_coef * out_dict['reg']
        
        return {'loss': loss, 'R_loss': R_loss}

    def get_loss_observation(self):
        return ['loss', 'R_loss']
        
        
        