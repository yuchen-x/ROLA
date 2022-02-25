from sacred import Experiment, observers

ex = Experiment("rola")


@ex.config
def default_config():
    # environment name: str
    env_name = "CT"
    # number of parallel envs: int
    n_envs = 1
    # total training episodes: int
    total_epies = 100000
    # horizon of each episode: float
    max_epi_steps = 25
    # discount factor: float
    gamma = 0.95
    # actor's learning rate: float
    a_lr = 1e-3
    # critic's learning rate: float
    c_lr = 1e-3
    # number of iterations for training local critic: int
    local_c_train_iteration = 4
    # epsilong start value: float
    eps_start = 1.0
    # epsilong end value: float
    eps_end = 0.01
    # epsilon decay period: float
    eps_decay_epis = 15000
    # training frequency (episode): int
    train_freq = 2
    # target net updating frequency (episode): int
    c_target_update_freq = 16
    # whether use soft target net update or not: bool
    c_target_soft_update = False
    # learning rate for target net soft updating: float
    tau = 0.01
    # whether use Reinforce with baseline or not: bool
    MC_baseline = False
    # n-step for n-step TD (0 and 1 both means regualr TD): int
    n_step_bootstrap = 0
    # network architecture for actor and critic: int
    a_mlp_layer_size = 64
    a_rnn_layer_size = 64
    cen_c_mlp_layer_size = 64
    local_c_mlp_layer_size = 64
    # gradient clipping: None/float
    grad_clip_value = None
    grad_clip_norm = 1.0
    # whether evaluate policy or not: bool
    eval_policy = False
    # Evaluate the learned policy every 100 episodes: int
    eval_freq = 100
    # Evaluate the policy over 10 episodes: int
    eval_num_epi = 10
    # whether obs last action or not: bool
    obs_last_action = False
    # particle env params
    prey_accel = 4.0
    prey_max_v = 1.3
    obs_r = 1.0
    obs_resolution = 8
    flick_p = 0.0
    enable_boundary = False
    benchmark = False
    # discretize continuous action space in OpenAI particle envs
    # 1 - 5 (up, down, left, right, stay)
    # 2 - 9 (8 directions + stay)
    discrete_mul = 1
    # scenario name for Antipodal Navigation particle env: str
    config_name = "cross"
    # other env params
    grid_dim = [4, 4]
    target_rand_move = False
    n_target = 1
    small_box_only = False
    terminal_reward_only = False
    big_box_reward = 100
    small_box_reward = 10
    n_agent = 2
    # the id of one run: int
    run_idx = 0
    # saving frequency (episode) for checkpoint and results: int
    save_rate = 1000
    # the name of the directory to save results: str
    save_dir = "trial"
    # whether save checkpoint or not: bool
    save_ckpt = False
    # when to save checkpoint (hr): int
    save_ckpt_time = 23
    # whether continuing training from a checkpoint or not: bool
    resume = False


@ex.named_config
def MC_baseline():
    MC_baseline = True


@ex.named_config
def c_target_soft_update():
    c_target_soft_update = True


@ex.named_config
def no_discnt_a_loss():
    discnt_a_loss = False


@ex.named_config
def eval_policy():
    eval_policy = True


@ex.named_config
def enable_boundary():
    enable_boundary = True


@ex.named_config
def obs_last_action():
    obs_last_action = True


@ex.named_config
def benchmark():
    benchmark = True


@ex.named_config
def target_rand_move():
    target_rand_move = True


@ex.named_config
def small_box_only():
    small_box_only = True


@ex.named_config
def terminal_reward_only():
    terminal_reward_only = True


@ex.named_config
def save_ckpt():
    save_ckpt = True


@ex.named_config
def resume():
    resume = True
