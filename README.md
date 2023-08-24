# GFN4Rec

GFlowNet for listwise recommendation.

## 0.Setup

```
conda create -n gfn4rec python=3.8
conda activate gfn4rec
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn tqdm ipykernel
python -m ipykernel install --user --name gfn4rec --display-name "GFN4Rec"
```

## 1. User Response Model

#### 1.1 [KuaiRand-1K](https://kuairand.com/)

```
@inproceedings{gao2022kuairand,
  title = {KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos},
  author = {Chongming Gao and Shijun Li and Yuan Zhang and Jiawei Chen and Biao Li and Wenqiang Lei and Peng Jiang and Xiangnan He},
  url = {https://doi.org/10.1145/3511808.3557624},
  doi = {10.1145/3511808.3557624},
  booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  series = {CIKM '22},
  location = {Atlanta, GA, USA},
  numpages = {5},
  year = {2022}
}
```

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

#### 1.3 [ContentWise](https://github.com/ContentWise/contentwise-impressions)

```
@inproceedings{contentwise-impressions,
 author = {P\'{e}rez Maurera, Fernando B. and Ferrari Dacrema, Maurizio and Saule, Lorenzo and Scriminaci, Mario and Cremonesi, Paolo},
 title = {ContentWise Impressions: An Industrial Dataset with Impressions Included},
 year = {2020},
 isbn = {9781450368599},
 publisher = {Association for Computing Machinery},
 address = {New York, NY, USA},
 url = {https://doi.org/10.1145/3340531.3412774},
 doi = {10.1145/3340531.3412774},
 booktitle = {Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management},
 pages = {3093–3100},
 numpages = {8},
 keywords = {dataset, implicit feedback, impressions, collaborative filtering, open source},
 location = {Virtual Event, Ireland},
 series = {CIKM '20}
}
```





# 2. Simulated Environment Example

> KuaiRand Multi-Behavior Simulator.ipynb

# 3. Train RL Agent

### 3.1 GFN

#### Detailed Balance loss:

```
bash train_gfn_db_XXX.sh
```
Note XXX represents the corresponding dataset name (e.g. movielens, kuairand, contentwise)

#### Trajectory Balance loss:

```
bash train_gfn_tb_XXX.sh
```

### 3.2 Learning-to-Rank Baselines:

Pointwise ranker:

```
bash train_ptranker_XXX.sh
```

### 3.3 List-wise Ranking Baselines:

#### PRM


#### GRN


### 3.4 Generative Rec Baselines:

#### ListCVAE


#### PivotCVAE


### 3.5 RL Baselines:

#### TD3


# 4. Result Observation

Training curves check:

> TrainingObservation.ipynb

