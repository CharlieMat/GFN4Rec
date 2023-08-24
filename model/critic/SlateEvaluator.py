import torch.nn.functional as F
import torch.nn as nn
import torch

from model.components import DNN
from utils import get_regularization

class SlateEvaluator(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - evaluator_lstm_hidden_dim
        - evaluator_lstm_n_layer
        - evaluator_mlp_hidden_dims
        - evaluator_dropout_rate
        '''
        parser.add_argument('--evaluator_lstm_hidden_dim', type=int, default=16, 
                            help='hidden size of BiLSTM')
        parser.add_argument('--evaluator_lstm_n_layer', type=int, default=1, 
                            help='number of layers in BiLSTM')
        parser.add_argument('--evaluator_mlp_hidden_dims', type=int, nargs='+', default=[128], 
                            help='hidden size of MLP kernel')
        parser.add_argument('--evaluator_dropout_rate', type=float, default=0.1, 
                            help='dropout rate of evaluator')
        return parser
    
    def __init__(self, args, state_dim, enc_dim):
        super().__init__()
        self.user_dim = state_dim
        self.item_dim = enc_dim
        self.h_dim = args.evaluator_lstm_hidden_dim
        self.intentEvolution = nn.LSTM(input_size=enc_dim,
                                       hidden_size=args.evaluator_lstm_hidden_dim,
                                       num_layers=args.evaluator_lstm_n_layer,
                                       batch_first=True, dropout=0, bidirectional=True)
        self.QNet = DNN(2*enc_dim + 2*args.evaluator_lstm_hidden_dim + state_dim, 
                        args.evaluator_mlp_hidden_dims, 1, 
                        dropout_rate = args.evaluator_dropout_rate, do_batch_norm = True)
        
    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {
            'state': (B, state_dim)
            'action': (B, slate_size, enc_dim)}
        '''
        # (B, state_dim)
        state_emb = feed_dict['state'].view(-1, 1, self.user_dim)
        B = state_emb.shape[0]
        # (B, slate_size, enc_dim)
        V = feed_dict['action'].view(B, -1, self.item_dim)
        K = V.shape[1]
        
        # Intent Evolution
        # (B, slate_size, hidden_dim * 2)
        O, _ = self.intentEvolution(V)
        
        # Mutual influence
        # (B, slate_size, slate_size)
        attn = torch.sum(V.view(B,K,1,self.item_dim) * V.view(B,1,K,self.item_dim), dim = 3)
        attn = torch.softmax(attn, dim = 2)
        # (B, slate_size, enc_dim)
        A = torch.sum(attn.view(B,K,K,1) * V.view(B,K,1,self.item_dim), dim = 2)
        
        # Final score
        # (B, slate_size, state_dim + hidden_dim * 2 + enc_dim * 2)
        X = torch.cat((state_emb.tile([1,K,1]), O, V, A), dim = 2)
        # (B, slate_size)
        Q = self.QNet(X).view(B,K)
        
        reg = get_regularization(self.intentEvolution, self.QNet)
        return {'q': Q, 'reg': reg}