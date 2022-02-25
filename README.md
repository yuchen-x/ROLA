# ROLA: Robust Local Advantage Actor-Critic

This is the code for implementing the ROLA algorithm presented in the paper:
[Local Advantage Actor-Critic for Robust Multi-Agent Deep Reinforcement Learning](https://arxiv.org/pdf/2110.08642.pdf), IEEE The 3rd International Symposium on Multi-Robot and Multi-Agent Systems (MRS), 2021. **Best Paper Award Finalist**.

# Minimum Requirements 

- Python 3.7.5
- Pytorch 1.4.0
- OpenAI Gym 0.15.4
- Sacred 0.8.1

# Installation

- To reproduce the results presented in the paper, all dependecies can be installed by running

```
cd ROLA/Anaconda_Env/
conda env create -f mrs21.yml
```

- To install all modules of the algorithm and domains

```
cd ROLA
pip install -e .
```

# Core Hyper-Parameter

- `a_lr`, learning rate for actor updates (default: `1e-3`)
- `c_lr`, learning rate for critic updates (default: `1e-3`)
- `local_c_train_iteration`, the number of training iterations for updating local critic (default: `4`)
- `eps_start`, the initial value for epsilon decay (default: `1.0`)
- `eps_end`, the ending value for epsilon decay (default: `0.01`)
- `eps_decay_epis`, the period of epslion decay
- `train_freq`, perform trainning very # of episodes (default: `2`)
- `n_envs`, the number of parallel envs (default: equal to the `train_freq`)
- `c_target_update_freq`, update the target-net of the critics every # of episodes (default: `16`) 
- `n_step_bootstrap`, n-step TD (default: `3`)
- `grad_clip_norm`, gradient clipping (default: `1.0`)

# How to Run

- Capture Target Domain (Grid World 6x6)

```
rola.py with env_name='CT' n_agent=2 grid_dim=[6,6] n_envs=2 max_epi_steps=60 a_lr=0.0005 c_lr=0.0005 local_c_train_iteration=1 train_freq=2 c_target_update_freq=16 n_step_bootstrap=3 eps_decay_epis=15000 eps_end=0.05 total_epies=80000 eval_num_epi=10 eval_freq=100 eval_policy save_ckpt save_dir='ROLA_CT_6' run_idx=0 
```

- Small Box Pushing Domain (Grid World 6x6)

```
rola.py with env_name='SBP' n_agent=2 grid_dim=[6,6] n_envs=2 max_epi_steps=100 small_box_reward=100 gamma=0.98 small_box_only terminal_reward_only a_lr=0.001 c_lr=0.003 local_c_train_iteration=4 train_freq=2 c_target_update_freq=32 n_step_bootstrap=3 eps_decay_epis=2000 eps_end=0.01 grad_clip_norm='None' total_epies=4000 eval_num_epi=10 eval_freq=100 eval_policy save_ckpt save_dir='ROLA_BP_6' run_idx=0
```

- Cooperative Navigation (observation radius 1.4)

```
rola.py with env_name='pomdp_simple_spread' n_agent=3 n_envs=2 max_epi_steps=25 discrete_mul=2 obs_r=1.4 a_lr=0.001 c_lr=0.001 local_c_train_iteration=4 train_freq=2 c_target_update_freq=16 n_step_bootstrap=5 eps_decay_epis=50000 eps_end=0.05 total_epies=100000 eval_num_epi=10 eval_freq=100 eval_policy save_ckpt save_dir='ROLA_CN_6' run_idx=0
```

- Antipodal Navigation (observation radius 1.4)

```
rola.py with env_name='pomdp_advanced_spread' config_name='antipodal' n_agent=4 n_envs=2 max_epi_steps=50 discrete_mul=1 obs_r=1.4 a_lr=0.001 c_lr=0.001 local_c_train_iteration=4 train_freq=2 c_target_update_freq=16 n_step_bootstrap=5 eps_decay_epis=20000 eps_end=0.05 total_epies=30000 eval_num_epi=10 eval_freq=100 eval_policy save_ckpt save_dir='ROLA_AN_6' run_idx=0
```

# Code Structure
- `./scripts/rola.py` the main training loop of ROLA
- `./src/pg_marl/rola/` all source code of ROLA
- `./src/pg_marl/rola/learner.py` the core code for ROLA algorithm
- `./src/marl_envs/` all source code of the domains considered in the paper

# Citation

If you used this code for your research or found it helpful, please consider citing this paper:

<pre>
@InProceedings{xiao_mrs_2021,
  author = "Xiao, Yuchen and Lyu, Xueguang and Amato, Christopher",
  title = "Local Advantage Actor-Critic for Robust Multi-Agent Deep Reinforcement Learning",
  booktitle = "IEEE The 3rd International Symposium on Multi-Robot and Multi-Agent Systems",
  year = "2021"
}
</pre>
