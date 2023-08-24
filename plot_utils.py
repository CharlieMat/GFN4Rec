import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from tqdm import tqdm

def get_online_training_info(log_path, episode_features = [], training_losses = []):
    episode = []
    episode_report_list = {k: [] for k in episode_features}
    loss_report_list = {k: [] for k in training_losses}
    with open(log_path, 'r') as infile:
        args = eval(infile.readline())
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("online episode:"):])
            if len(episode_report_list) == 0:
                episode_report_list = {k:[v] for k,v in episode_report.items()}
            else:
                for k,L in episode_report_list.items():
                    L.append(episode_report[k])
            # loss report
            loss_report = eval(split[2].strip()[len("training:"):])
            if len(loss_report_list) == 0:
                loss_report_list = {k:[v] for k,v in loss_report.items()}
            else:
                for k,L in loss_report_list.items():
                    L.append(loss_report[k])
    info = {'episode': episode}
    info.update(episode_report_list)
    info.update(loss_report_list)
    return info

def get_offline_test_info(log_path):
    with open(log_path, 'r') as infile:
        test_result = infile.readline()
        print(test_result)
    return test_result

def smooth(values, window = 3):
    half_window = max(0,window//2)
    new_values = [np.mean(values[max(0,idx-half_window):min(idx+half_window+1,len(values))]) for idx in range(len(values))]
    return new_values

def multiplot_multiple_lines(legend_names, list_of_stats, x_name, ncol = 2, row_height = 4):
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [{field_name: [values]}]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    plt.rcParams.update({'font.size': 14})
    assert ncol > 0
    features = list(list_of_stats[0].keys())
    features.remove(x_name)
    N = len(features)
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            X = list_of_stats[j][x_name]
            value_list = list_of_stats[j][field]
            minY,maxY = min(minY,min(value_list)),max(maxY,max(value_list))
            plt.plot(X[:len(value_list)], value_list, label = L)
        plt.ylabel(field)
        plt.xlabel(x_name)
        scale = 1e-4 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        plt.legend()
    plt.show()

def plot_multiple_lines(list_of_stats, labels, 
                        fig_height = 4, font_size = 16, log_value = False):
    '''
    @input:
    - list_of_stats: [[x],[y]]
    - labels: [title_name]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    plt.figure(figsize = (16, fig_height))
    for i,stats in enumerate(list_of_stats):
        X,Y = stats
        plt.plot(X,Y,label = labels[i])
        if log_value:
            plt.yscale('log')
        plt.title(labels[i])
    plt.legend()
    plt.show()
    
def plot_multiple_bars(list_of_stats, features, 
                       ncol = 2, row_height = 4, font_size = 16, 
                       log_value = False, horizontal = False):
    '''
    @input:
    - list_of_stats: [[x],[x_name],[y]]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    assert ncol > 0 and len(features) == N
    fig_height = 12 // ncol if N == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))

    for i,stats in enumerate(list_of_stats):
        X,X_name,Y = stats
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        if horizontal:
            plt.barh(X,np.log(Y) if log_value else Y,label = features[i])
            plt.yticks(X,X_name)
            if log_value:
                plt.xlabel('freq in log')
        else:
            plt.bar(X,np.log(Y) if log_value else Y,label = features[i])
            plt.xticks(X,X_name)
            if log_value:
                plt.ylabel('freq in log')
        plt.title(features[i])
    plt.show()
    
def plot_multiple_hists(list_of_stats, features, 
                        ncol = 2, row_height = 4, font_size = 16, 
                        log_value = False, n_bin = 10):
    '''
    @input:
    - list_of_stats: [[y]]
    - ncol: number of subplots in each row
    - row_height: height of each row
    '''
    plt.rcParams.update({'font.size': font_size})
    N = len(list_of_stats)
    assert ncol > 0 and len(features) == N
    fig_height = 12 // ncol if N == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,Y in enumerate(list_of_stats):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        plt.hist(Y,label = features[i], bins = n_bin)
        plt.title(features[i])
        if log_value:
            plt.yscale('log')
    plt.show()
    
def plot_mean_var_line(legend_names, list_of_stats, x_name,  ncol = 2, row_height = 4, window = None, 
                       save_path = "", font_size = 14):        
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [[{field_name: [values]}]]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    color_lib = [("r", "salmon"), ("g", "springgreen"), ("b", "dodgerblue"), ("y", "lightyellow"), ('black', 'lightgrey'), ('purple', 'magenta')]
    plt.rcParams.update({'font.size': font_size})
    features = list(list_of_stats[0][0].keys())
    features.remove(x_name)
    seeds_len = list(list_of_stats[0])
    N = len(features)
    X = list_of_stats[0][0][x_name]
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            mean_map = [[] for _ in range(len(X))]
            for seed in range(len(list_of_stats[j])):
                for k, v in enumerate(list_of_stats[j][seed][field]):
                    mean_map[k].append(v)
            mean_curve = []
            up_curve = []
            down_curve = []
            half = len(list_of_stats[0]) // 2
            if len(list_of_stats[0]) != 1:
                for v in mean_map:
                    mean_curve.append(np.mean(v))
                    down_curve.append(np.mean(sorted(v)[:half]))
                    up_curve.append(np.mean(sorted(v)[len(list_of_stats[0]) - half:]))
            else:
                for v in mean_map:
                    mean_curve.append(np.mean(v))
                    down_curve = mean_curve
                    up_curve = mean_curve
            if window:
                mean_curve = smooth(mean_curve, window)
                down_curve = smooth(down_curve, window)
                up_curve = smooth(up_curve, window)
            mean_curve = np.array(mean_curve)
            up_curve = np.array(up_curve)
            down_curve = np.array(down_curve)
            minY,maxY = min(down_curve.min(), minY), max(up_curve.max(), maxY)
            plt.plot(X, mean_curve, color=color_lib[j % len(color_lib)][0], linewidth=1.0, label=L)
            plt.fill_between(X, up_curve, down_curve, facecolor=color_lib[j % len(color_lib)][1], alpha=0.3)
        plt.ylabel(field)
#         plt.title(field)
        plt.xlabel(x_name)
        scale = 1e-4 + maxY - minY
        try:
            plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        except:
            print('ylim:', minY, maxY, scale)
        plt.legend()
    if len(save_path) > 0:
        plt.savefig(save_path, format='svg', dpi=500)
    
    plt.show()