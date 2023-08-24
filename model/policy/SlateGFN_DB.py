import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class SlateGFN_DB(BaseOnlinePolicy):
    '''
    GFlowNet with Detailed Balance for listwise recommendation
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - gfn_forward_hidden_dims
        - gfn_flow_hidden_dims
        - gfn_forward_offset
        - gfn_reward_smooth
        - gfn_Z
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
        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of state_slate encoding layers')
        parser.add_argument('--gfn_flow_hidden_dims', type=int, nargs="+", default=[128], 
                            help='hidden dimensions of flow estimator')
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, 
                            help='smooth offset of forward logp of TB loss')
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, 
                            help='reward smooth offset in the backward part of TB loss')
        parser.add_argument('--gfn_Z', type=float, default=0., 
                            help='average reward offset')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        # BaseModel initialization: 
        # - reader_stats, model_path, loss_type, l2_coef, no_reg, device, slate_size
        # - _define_params(args)
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flow_hidden_dims = args.gfn_flow_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset
        self.gfn_reward_smooth = args.gfn_reward_smooth
        self.gfn_Z = args.gfn_Z
        super().__init__(args, reader_stats, device)
        self.display_name = "GFN_DB"
        
    def to(self, device):
        new_self = super(SlateGFN_DB, self).to(device)
        return new_self

    def _define_params(self, args):
        # userEncoder, enc_dim, state_dim, bce_loss
        super()._define_params(args)
        # p_forward
        self.pForwardEncoder = DNN(self.state_dim + self.enc_dim * self.slate_size, 
                                   args.gfn_forward_hidden_dims, self.enc_dim, 
                                   dropout_rate = args.dropout_rate, do_batch_norm = True)
        self.pForwardNorm = nn.LayerNorm(self.enc_dim)
        # flow
        self.logFlow = DNN(self.state_dim + self.enc_dim * self.slate_size, 
                           args.gfn_flow_hidden_dims, 1, 
                           dropout_rate = args.dropout_rate, do_batch_norm = True)

    
    def generate_action(self, user_state, feed_dict):
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        parent_slate = feed_dict['action'] # (B, K)
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        '''
        @input:
        - user_state: (B, state_dim) 
        - feed_dict: same as BaseOnlinePolicy.get_forward@feed_dict
        @output:
        - out_dict: {'prob': (B, K), 
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
            
        # (1,L,enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(candidates['item_id'], 
                                                       {k[5:]: v for k,v in candidates.items() if k != 'item_id'}, 
                                                                     B if batch_wise else 1)
        
        # forward probabilities P(a_t|s_{t-1}) of the action, size (B, K)
        current_P = torch.zeros(B, slate_size).to(self.device)
        # (B, K)
        current_action = torch.zeros(B, slate_size).to(torch.long).to(self.device)
        # (B, K, enc_dim)
        current_list_emb = torch.zeros(B, slate_size, self.enc_dim).to(self.device)
        # (B, K+1)
        current_flow = torch.zeros(B, slate_size + 1).to(self.device)
        
        # regressive action generation
        for i in range(slate_size):
            # (B, state_dim + slate_size * enc_dim)
            current_state = torch.cat((user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), dim = 1)
            # (B, enc_dim)
            selection_weight = self.pForwardEncoder(current_state)
            selection_weight = self.pForwardNorm(selection_weight)
            # (B, L)
            score = torch.sum(selection_weight.view(B,1,self.enc_dim) * candidate_item_enc, dim = -1) #/ self.enc_dim
            # (B, L)
            prob = torch.softmax(score, dim = 1)

            # (B,)
            if is_train or torch.is_tensor(parent_slate):
                # during training, output the target action probability without sampling
                action_at_i = parent_slate[:,i] # (B,)
                current_P[:,i] = torch.gather(prob,1,action_at_i.view(-1,1)).view(-1)
                current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[action_at_i]
                current_flow[:,i] = self.logFlow(current_state).view(-1)
                current_action[:,i] = action_at_i
            else:
                if i > 0:
                    # remove items already selected
                    prob.scatter_(1,current_action[:,:i],0)
             
                if do_explore:
                    # exploration: categorical sampling or uniform sampling
                    if do_uniform:
                        indices = Categorical(torch.ones_like(prob)).sample()
                    else:
                        indices = Categorical(prob).sample()
                else: 
                    # greedy: topk selection
                    _, indices = torch.topk(prob, k = 1, dim = 1)
                indices = indices.view(-1).detach()
                # update current slate action
                current_action[:,i] = indices
                # update slate action probability
                current_P[:,i] = torch.gather(prob,1,indices.view(-1,1)).view(-1)

                if batch_wise:
                    for j in range(B):
                        current_list_emb[j,i,:] = candidate_item_enc[j,indices[j]]
                else:
                    current_list_emb[:,i,:] = candidate_item_enc.view(-1,self.enc_dim)[indices]
        if is_train:
            # the terminal flow
            current_state = torch.cat((user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), dim = 1)
            current_flow[:,-1] = self.logFlow(current_state).view(-1)
            # regularization
            reg = self.get_regularization(self.logFlow, self.pForwardEncoder)
        else:
            reg = 0

        out_dict = {'prob': current_P, 
                    'action': current_action, 
                    'logF': current_flow, 
                    'reg': reg}
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        Detailed balance loss (Note: log(P(s[t-1]|s[t])) = 0 if tree graph): 
        * non-terminal: ( log(flow(s[t-1])) + log(P(s[t]|s[t-1])) - log(flow(s[t])) )^2
        * terminal: ( log(flow(s[t])) - log(reward(s[t])) )^2
        
        @input:
        - feed_dict: same as BaseOnlinePolicy.get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'logF': (B,)
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K*n_feedback),
            'reward': (B,)}
        @output
        - loss
        '''
        # (B, K)
        parent_flow = out_dict['logF'][:,:-1]
        # (B, K)
        current_flow = out_dict['logF'][:,1:]
        # (B, K)
        log_P = torch.log(out_dict['prob'] + self.gfn_forward_offset)
        # (B, K)
        forward_part = parent_flow + log_P + self.gfn_Z
        # (B, K)
        backward_part = current_flow
        # scalar
        DB_loss = torch.mean((forward_part - backward_part).pow(2))
        # (B, )
        terminal_loss = (current_flow[:,-1] + self.gfn_Z \
                         - torch.log(out_dict['reward'] + self.gfn_reward_smooth + 1e-6).view(-1) ).pow(2)
        terminal_loss = torch.mean(terminal_loss)
        # scalar
        loss = DB_loss + terminal_loss + self.l2_coef * out_dict['reg']
        
        return {'loss': loss, 'DB_loss': DB_loss, 'terminal_loss': terminal_loss, 
                'forward_part': torch.mean(forward_part), 'backward_part': torch.mean(backward_part), 
                'prob': torch.mean(out_dict['prob'])}

    def get_loss_observation(self):
        return ['loss', 'DB_loss', 'terminal_loss', 'forward_part', 'backward_part', 'prob']
        
        
        