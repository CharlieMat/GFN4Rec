# GFN4Rec

GFlowNet for listwise recommendation.

Citation:
```
@inproceedings{10.1145/3580305.3599364,
author = {Liu, Shuchang and Cai, Qingpeng and He, Zhankui and Sun, Bowen and McAuley, Julian and Zheng, Dong and Jiang, Peng and Gai, Kun},
title = {Generative Flow Network for Listwise Recommendation},
year = {2023},
isbn = {9798400701030},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3580305.3599364},
doi = {10.1145/3580305.3599364},
booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1524–1534},
numpages = {11},
keywords = {recommender systems, online learning, generative model},
location = {Long Beach, CA, USA},
series = {KDD '23}
}
```

## 0.Setup

```
conda create -n gfn4rec python=3.8
conda activate gfn4rec
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
pip install -r requirements.txt
python -m ipykernel install --user --name gfn4rec --display-name "GFN4Rec"
```

## 1. User Response Model

#### 1.1 [KuaiRand-1K](https://kuairand.com/)

First check KuaiRandPreprocess.ipynb for preprocessing steps and KuaiRandDataset.ipynb for dataset details.

Data format: 
> (user_id, video_id, timestamp, is_click, is_like, is_comment, is_forward, is_follow, is_hate, long_view)

Video meta data format: 
> (video_id, video_type, upload_type, music_type, log_duration, tag)

User meta data format: 
> (user_active_degree, is_live_streamer, is_video_author, follow_user_num_range, fans_user_num_range, friend_user_num_range, register_days_range, onehot_feat{0,1,6,9,10,11,12,13,14,15,16,17})

Then run script for user response model pretraining:
```
bash train_multi_behavior_user_response_kuairand.sh
```

Resulting model and log will be saved in 'output/kuairand/env/'

Note: multi-behavior user response models consists the state_encoder that is assumed to be the ground truth user state transition model.

#### 1.2 [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/)

```
https://grouplens.org/datasets/movielens/1m/
```

First check ML1MPreprocess.ipynb for preprocessing steps.

Processed data format:
> (user_id, movie_id, timestamp, is_click, is_like, is_star)

Movie meta data format:
> (movie_id, genres)

User meta data format:
> (user_id, gender, age)

Then run script for user response model pretraining:
```
bash train_multi_behavior_user_response_ml1m.sh
```

Resulting model and log will be saved in *output/ml1m/env/*


# 2. Simulated Environment Example

> KuaiRand Multi-Behavior Simulator.ipynb

# 3. Train Online/Offline Agents

All related training scripts in *scripts/*

### Model Implimentation

All models are implemented in *model/*

* model/agent: the training agents that coordinate the learning of different components
* model/buffer: the experience replay buffer for online training
* model/policy: the recommendation model
* model/simulator: the user response model integrated in the simulated environment

All model training scripts are named as:

```
train_<model_name>_<dataset>.sh
```

For example:

```
bash train_gfn_db_kuairand.sh
bash train_gfn_tb_kuairand.sh
```

Output logs and model checkpoints will be saved in *output/<dataset>/agent/*

# 4. Result Observation

For training curves check:

> TrainingObservation.ipynb

For offline training observation, check:

> OfflineObservation.ipynb

For GFN4Rec analysis, check:

> GFN Observation.ipynb
