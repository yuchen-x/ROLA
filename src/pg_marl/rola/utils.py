import torch
import random
import numpy as np
import pickle


class Agent:
    """
    A class used to represent an agent
    """

    def __init__(self):
        self.idx = None
        self.actor_net = None
        self.acotr_tgt_net = None
        self.actor_optimizer = None
        self.actor_loss = None

        self.critic_net = None
        self.critic_tgt_net = None
        self.critic_optimizer = None
        self.critic_loss = None


class LinearDecay(object):
    """
    A class used for computing a linear decaying value

    """

    def __init__(self, total_steps, init_value, end_value):
        """
        Parameters
        ----------
        total_epies: int
            the period to decay
        init_value: float
            the initial value
        end_value: float
            the ending value
        """

        self.total_steps = total_steps
        self.init_value = init_value
        self.end_value = end_value

    def get_value(self, step):
        """
        Compute the value at the current step

        Parameters
        ----------
        step: int
            current decaying step

        Return
        ------
        float
            a float number
        """
        frac = min(float(step) / self.total_steps, 1.0)
        return self.init_value + frac * (self.end_value - self.init_value)


def save_train_data(run_idx, data, save_dir):
    """
    Save training data

    Parameters:
    ----------
    data: List[float]
        a list of training returns
    save_dir: str
        name of a saving directory
    """

    with open(
        "./performance/" + save_dir + "/train/train_perform" + str(run_idx) + ".pickle", "wb"
    ) as handle:
        pickle.dump(data, handle)


def save_test_data(run_idx, data, save_dir):
    """
    Save testing data

    Parameters:
    ----------
    data: List[float]
        a list of testing returns
    save_dir: str
        name of a saving directory
    """

    with open(
        "./performance/" + save_dir + "/test/test_perform" + str(run_idx) + ".pickle", "wb"
    ) as handle:
        pickle.dump(data, handle)


def save_checkpoint(run_idx, epi_count, controller, learner, envs_runner, save_dir):
    """
    Save checkpoint

    Parameters:
    ----------
    run_idx: int
        index of a run
    epi_count: int
        the number of episodes that have done
    controller: MAC
        an instance of MAC class (see ./src/pg_marl/rola/controller.py)
    learner: Learner
        an instance of Learner class (see ./src/pg_marl/rola/learner.py)
    envs_runner: EnvsRunner
        an instance of EnvsRunner class (see ./src/pg_marl/rola/envs_runner.py)
    save_dir: str
        name of a saving directory
    """

    # set saving path
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "1.tar"
    # save genric configuration info
    torch.save(
        {
            "epi_count": epi_count,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "envs_runner_returns": envs_runner.returns,
            "controller_eval_returns": controller.eval_returns,
            "smoothed_return": envs_runner.smoothed_return,
            "cen_critic_net_state_dict": learner.cen_critic_net.state_dict(),
            "cen_critic_tgt_net_state_dict": learner.cen_critic_tgt_net.state_dict(),
            "cen_critic_optimizer_state_dict": learner.cen_critic_optimizer.state_dict(),
        },
        PATH,
    )
    # request rand state info of each thread
    for idx, parent in enumerate(envs_runner.parents):
        parent.send(("get_rand_states", None))
    # receive and save rand state info of each thread
    for idx, parent in enumerate(envs_runner.parents):
        PATH = (
            "./performance/"
            + save_dir
            + "/ckpt/"
            + str(run_idx)
            + "_env_rand_states_"
            + str(idx)
            + "1.tar"
        )
        rand_states = parent.recv()
        torch.save(rand_states, PATH)
    # save each agent's attributes
    for idx, agent in enumerate(controller.agents):
        PATH = (
            "./performance/"
            + save_dir
            + "/ckpt/"
            + str(run_idx)
            + "_agent_"
            + str(idx)
            + "1.tar"
        )
        torch.save(
            {
                "actor_net_state_dict": agent.actor_net.state_dict(),
                "actor_tgt_net_state_dict": agent.actor_tgt_net.state_dict(),
                "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
                "critic_net_state_dict": agent.critic_net.state_dict(),
                "critic_tgt_net_state_dict": agent.critic_tgt_net.state_dict(),
                "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            },
            PATH,
        )


def load_checkpoint(run_idx, controller, learner, envs_runner, save_dir):
    """
    Load checkpoint

    Parameters:
    ----------
    run_idx: int
        index of a run
    controller: MAC
        an instance of MAC class (see ./src/pg_marl/rola/controller.py)
    learner: Learner
        an instance of Learner class (see ./src/pg_marl/rola/learner.py)
    envs_runner: EnvsRunner
        an instance of EnvsRunner class (see ./src/pg_marl/rola/envs_runner.py)
    save_dir: str
        name of a saving directory

    Return:
    -------
    epi_count: int
        the number of episodes that have done
    """

    # load generic stuff
    PATH = "./performance/" + save_dir + "/ckpt/" + str(run_idx) + "_genric_" + "1.tar"
    ckpt = torch.load(PATH)
    epi_count = ckpt["epi_count"]
    random.setstate(ckpt["random_state"])
    np.random.set_state(ckpt["np_random_state"])
    torch.set_rng_state(ckpt["torch_random_state"])
    envs_runner.returns = ckpt["envs_runner_returns"]
    controller.eval_returns = ckpt["controller_eval_returns"]
    envs_runner.smoothed_return = ckpt["smoothed_return"]
    learner.cen_critic_net.load_state_dict(ckpt["cen_critic_net_state_dict"])
    learner.cen_critic_tgt_net.load_state_dict(ckpt["cen_critic_tgt_net_state_dict"])
    learner.cen_critic_optimizer.load_state_dict(ckpt["cen_critic_optimizer_state_dict"])

    # load random states for all workers
    for idx, parent in enumerate(envs_runner.parents):
        PATH = (
            "./performance/"
            + save_dir
            + "/ckpt/"
            + str(run_idx)
            + "_env_rand_states_"
            + str(idx)
            + "1.tar"
        )
        rand_states = torch.load(PATH)
        parent.send(("load_rand_states", rand_states))

    # load actor and ciritc models for each agent
    for idx, agent in enumerate(controller.agents):
        PATH = (
            "./performance/"
            + save_dir
            + "/ckpt/"
            + str(run_idx)
            + "_agent_"
            + str(idx)
            + "1.tar"
        )
        ckpt = torch.load(PATH)
        agent.actor_net.load_state_dict(ckpt["actor_net_state_dict"])
        agent.actor_tgt_net.load_state_dict(ckpt["actor_tgt_net_state_dict"])
        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        agent.critic_net.load_state_dict(ckpt["critic_net_state_dict"])
        agent.critic_tgt_net.load_state_dict(ckpt["critic_tgt_net_state_dict"])
        agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])

    return epi_count
