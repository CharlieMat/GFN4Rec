import torch

def get_immediate_reward(user_feedback):
    '''
    @input:
    - user_feedback: {'immediate_response': (B, slate_size, n_feedback), 
                      'immediate_response_weight': (n_feedback),
                      ... other feedbacks}
    @output:
    - reward: (B,)
    '''
    # (B, slate_size, n_feedback)
    if 'immediate_response_weight' in user_feedback:
        point_reward = user_feedback['immediate_response'] * user_feedback['immediate_response_weight'].view(1,1,-1)
    else:
        point_reward = user_feedback['immediate_response']
    # (B, slate_size)
    combined_reward = torch.sum(point_reward, dim = 2)
    # (B,)
    reward = torch.mean(combined_reward, dim = 1)
    return reward
    