import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.KRMBSeqReader import KRMBSeqReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab

class KRMBSlateReader(KRMBSeqReader):
    '''
    KuaiRand Multi-Behavior Data Reader
    '''
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        - from KRMBSeqReader:
            - user_meta_file
            - item_meta_file
            - max_hist_seq_len
            - val_holdout_per_user
            - test_holdout_per_user
            - meta_file_sep
            - from BaseReader:
                - train_file
                - val_file
                - test_file
                - n_worker
        '''
        parser = KRMBSeqReader.parse_data_args(parser)
        return parser
        
    def log(self):
        super().log()
        
    def __init__(self, args):
        '''
        - slate_size
        - from KRMBSeqReader:
            - max_hist_seq_len
            - val_holdout_per_user
            - test_holdout_per_user
            - from BaseReader:
                - phase
                - n_worker
        '''
        print("initiate KuaiRandMultiBehaior Slate reader")
        self.slate_size = args.slate_size
        super().__init__(args)
        
    def _sequence_holdout(self, args):
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        if args.val_holdout_per_user == 0 and args.test_holdout_per_user == 0:
            return {"train": self.log_data.index, "val": [], "test": []}
        data = {"train": [], "val": [], "test": []}
        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            n_train = len(sub_df) - (args.val_holdout_per_user + args.test_holdout_per_user) * self.slate_size
            data['train'].append(list(sub_df.index[:n_train])[::self.slate_size])
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user*self.slate_size]))
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user*self.slate_size::self.slate_size]))
        for k,v in data.items():
            data[k] = np.concatenate(v)
        return data
        
    def _read_data(self, args):
        '''
        - from KRMBSeqReader:
            - log_data: pd.DataFrame
            - data: {'train': [row_id], 'val': [row_id], 'test': [row_id]}
            - users: [user_id]
            - user_id_vocab: {user_id: encoded_user_id}
            - user_meta: {user_id: {feature_name: feature_value}}
            - user_vocab: {feature_name: {feature_value: one-hot vector}}
            - selected_user_features
            - items: [item_id]
            - item_id_vocab: {item_id: encoded_item_id}
            - item_meta: {item_id: {feature_name: feature_value}}
            - item_vocab: {feature_name: {feature_value: one-hot vector}}
            - selected_item_features: [feature_name]
            - padding_item_meta: {feature_name: 0}
            - user_history: {uid: [row_id]}
            - response_list: [response_type]
            - padding_response: {response_type: 0}
        - 
        '''
        super()._read_data(args)
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
            'user_id': (B,)
            'item_id': (B,slate_size) if train, (B,) otherwise
            'is_click', 'long_view', ...: (B,slate_size)
            'uf_{feature}': (B,F_dim(feature)), user features
            'if_{feature}': (B,slate_size,F_dim(feature))
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H, F_dim(feature))
            'history_{response}': (B, max_H)
            'loss_weight': (B, n_response)
        }
        '''
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]
        
        user_id = row['user_id'] # raw user ID
        user_meta = self.get_user_meta_data(user_id)
        
        # (slate_size,), {'if_{feature_name}': (slate_size * feature_dim,)}, {'if_{response_name}': (slate_size,)}
        item_id, item_meta, item_response = self.get_slate(user_id, row_id)
        record = {
            'user_id': self.user_id_vocab[row['user_id']], # encoded user ID
            'item_id': item_id # encoded item ID
        }
        record.update(user_meta)
        record.update(item_meta)
        record.update(item_response)
        
        # (max_H,)
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_hist_seq_len:]
        history, hist_length, hist_meta, hist_response = self.get_user_history(H_rowIDs)
        record['history'] = np.array(history)
        record['history_length'] = hist_length
        for f,v in hist_meta.items():
            record[f'history_{f}'] = v
        for f,v in hist_response.items():
            record[f'history_{f}'] = v
            
#         loss_weight = np.array([1. if record[f] == 1 else self.response_neg_sample_rate[f] \
#                                 for i,f in enumerate(self.response_list)])
#         record["loss_weight"] = loss_weight
        return record
    
    def get_slate(self, user_id, row_id):
        # ensure slate size alignment by setting: test_holdout_per_user + val_holdout_per_user >= 5
        S_rowIDs = [rid for rid in self.user_history[user_id] if row_id <= rid][:self.slate_size]
        
        H = self.log_data.iloc[S_rowIDs]
        # (slate_size,)
        slate_ids = np.array([self.item_id_vocab[iid] for iid in H['video_id']])
        # [{if_{feature}: one-hot vector}]
        meta_list = [self.get_item_meta_data(iid) for iid in H['video_id']] 
        # slate item meta features: {if_{feature_name}: (slate_size, feature_dim)}
        slate_meta = {} 
        for f in self.selected_item_features:
            # {if_{feature_name}: (slate_size, feature_dim)}
            slate_meta[f'if_{f}'] = np.array([v_dict[f'if_{f}'] for v_dict in meta_list])
        # {resp_type: (slate_size,)}
        slate_response = {}
        for resp in self.response_list:
            slate_response[resp] = np.array(H[resp])
        
        return slate_ids, slate_meta, slate_response

    def get_statistics(self):
        '''
        - slate_size
        - from KRMBSeqReader
            - raw_data_size
            - data_size
            - n_user
            - n_item
            - max_seq_len
            - user_features
            - user_feature_dims
            - item_features
            - item_feature_dims
            - feedback_type
            - feedback_size
            - feedback_negative_sample_rate
            - from BaseReader:
                - length
                - fields
        '''
        stats = super().get_statistics()
        stats["slate_size"] = self.slate_size
        return stats
