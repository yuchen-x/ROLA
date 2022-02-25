import gym
import torch
import random
import numpy as np
import os
import pickle
import time

from marl_envs.particle_envs.make_env import make_env
from marl_envs.my_env.capture_target import CaptureTarget as CT
from marl_envs.my_env.small_box_pushing import SmallBoxPushing as SBP

from pg_marl.rola.config import ex
from pg_marl.rola.envs_runner import EnvsRunner
from pg_marl.rola.memory import MemoryEpi
from pg_marl.rola.learner import Learner
from pg_marl.rola.controller import MAC
from pg_marl.rola.utils import (
    LinearDecay,
    save_train_data,
    save_test_data,
    save_checkpoint,
    load_checkpoint,
)

ENVIRONMENTS = {
    "CT": CT,
    "SBP": SBP,
}


@ex.main
def main(
    env_name,
    n_envs,
    total_epies,
    max_epi_steps,
    gamma,
    a_lr,
    c_lr,
    local_c_train_iteration,
    eps_start,
    eps_end,
    eps_decay_epis,
    train_freq,
    c_target_update_freq,
    c_target_soft_update,
    tau,
    MC_baseline,
    n_step_bootstrap,
    grad_clip_value,
    grad_clip_norm,
    a_mlp_layer_size,
    cen_c_mlp_layer_size,
    local_c_mlp_layer_size,
    a_rnn_layer_size,
    eval_policy,
    eval_freq,
    eval_num_epi,
    obs_last_action,
    prey_accel,
    prey_max_v,
    obs_r,
    obs_resolution,
    flick_p,
    enable_boundary,
    benchmark,
    discrete_mul,
    config_name,
    grid_dim,
    target_rand_move,
    n_target,
    small_box_only,
    terminal_reward_only,
    small_box_reward,
    big_box_reward,
    n_agent,
    seed,
    run_idx,
    save_rate,
    save_dir,
    save_ckpt,
    save_ckpt_time,
    resume,
):
    """
    The main function to run ROLA algorithm

    Parameters
    ----------
    see ./src/pg_marl/rola/config.py
    """

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    # create the dirs to save results
    os.makedirs("./performance/" + save_dir + "/train", exist_ok=True)
    os.makedirs("./performance/" + save_dir + "/test", exist_ok=True)
    os.makedirs("./performance/" + save_dir + "/ckpt", exist_ok=True)

    # collect params
    actor_params = {"a_mlp_layer_size": a_mlp_layer_size, "a_rnn_layer_size": a_rnn_layer_size}

    critic_params = {
        "cen_c_mlp_layer_size": cen_c_mlp_layer_size,
        "local_c_mlp_layer_size": local_c_mlp_layer_size,
    }

    hyper_params = {
        "gamma": gamma,
        "a_lr": a_lr,
        "c_lr": c_lr,
        "local_c_train_iteration": local_c_train_iteration,
        "c_target_update_freq": c_target_update_freq,
        "tau": tau,
        "grad_clip_value": grad_clip_value,
        "grad_clip_norm": grad_clip_norm,
        "MC_baseline": MC_baseline,
        "n_step_bootstrap": n_step_bootstrap,
    }

    particle_env_params = {
        "max_epi_steps": max_epi_steps,
        "prey_accel": prey_accel,
        "prey_max_v": prey_max_v,
        "obs_r": obs_r,
        "obs_resolution": obs_resolution,
        "flick_p": flick_p,
        "enable_boundary": enable_boundary,
        "benchmark": benchmark,
        "discrete_mul": discrete_mul,
        "config_name": config_name,
    }

    ct_params = {"terminate_step": max_epi_steps, "n_target": n_target, "n_agent": n_agent}

    box_pushing_params = {
        "terminate_step": max_epi_steps,
        "small_box_only": small_box_only,
        "terminal_reward_only": terminal_reward_only,
        "big_box_reward": big_box_reward,
        "small_box_reward": small_box_reward,
        "n_agent": n_agent,
    }

    # create env
    if env_name.startswith("CT"):
        ENV = ENVIRONMENTS[env_name]
        env = ENV(grid_dim=tuple(grid_dim), **ct_params)
    elif env_name.startswith("SBP"):
        ENV = ENVIRONMENTS[env_name]
        env = ENV(tuple(grid_dim), **box_pushing_params)
    else:
        env = make_env(env_name, discrete_action_input=True, **particle_env_params)
        env.seed(seed)

    # create memory buffer
    memory = MemoryEpi(env, obs_last_action=obs_last_action, size=train_freq)
    # cretate a controller for agents including all agents' policies
    controller = MAC(env, obs_last_action=obs_last_action, **actor_params)
    # create parallel envs runner
    envs_runner = EnvsRunner(
        env, n_envs, controller, memory, max_epi_steps, gamma, seed, obs_last_action
    )
    # create a learner with ROLA algorithm
    learner = Learner(env, env_name, controller, memory, **hyper_params, **critic_params)
    # create epsilon calculator for implementing e-greedy exploration policy
    eps_call = LinearDecay(eps_decay_epis, eps_start, eps_end)
    # count the number of episodes have done
    epi_count = 0
    # record time for saving checkpoint
    t_ckpt = time.time()
    # training starts from a check point
    if resume:
        epi_count = load_checkpoint(run_idx, controller, learner, envs_runner, save_dir)

    #################### Training loop ####################
    while epi_count < total_epies:

        # evaluate current policies
        if eval_policy and epi_count % (eval_freq - (eval_freq % train_freq)) == 0:
            controller.evaluate(gamma, max_epi_steps, eval_num_epi)
            print(
                f"{[run_idx]} Finished: {epi_count}/{total_epies} Evaluate learned policies with averaged returns {controller.eval_returns[-1]/n_agent} ...",
                flush=True,
            )

        # compute latest epsilon value
        eps = eps_call.get_value(epi_count)
        # rollout n episodes
        envs_runner.run(eps, n_epis=train_freq)
        # perform training
        learner.train(eps)
        epi_count += train_freq

        # update target-net parameters
        if c_target_soft_update:
            learner.update_critic_target_net(soft=True)
            learner.update_actor_target_net(soft=True)
        elif epi_count % c_target_update_freq == 0:
            learner.update_critic_target_net()
            learner.update_actor_target_net()
        # save training and testing results
        if epi_count % save_rate == 0:
            save_train_data(run_idx, envs_runner.returns, save_dir)
            save_test_data(run_idx, controller.eval_returns, save_dir)
        # save checkpoint
        if save_ckpt and (time.time() - t_ckpt) / 3600 >= save_ckpt_time:
            save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir)
            t_ckpt = time.time()
            break

    # save everthing in the end
    save_train_data(run_idx, envs_runner.returns, save_dir)
    save_test_data(run_idx, controller.eval_returns, save_dir)
    save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir)
    envs_runner.close()
    print("Finish entire training ... ", flush=True)


if __name__ == "__main__":
    ex.run_commandline()
