import numpy as np
import torch
import torch.nn.functional as F
import random

from multiprocessing import Process, Pipe


def worker(child, env, gamma, seed):
    """
    Worker function which interacts with the environment over remote

    Parameters
    ----------
    env: gym.env
        a gym environment
    gamma: float
        a discount factor
    seed: int
        a random seed
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        while True:
            # wait cmd sent by parent
            cmd, data = child.recv()
            if cmd == "step":
                obs, reward, terminate, _ = env.step(data)

                state = env.get_state()
                # sent experience back
                child.send(
                    (last_state, last_obs, data, reward, state, obs, terminate, gamma ** step)
                )

                last_obs = obs
                last_state = state
                R += gamma ** step * sum(reward) / env.n_agent
                step += 1

            elif cmd == "get_return":
                child.send(R)

            elif cmd == "reset":
                last_obs = env.reset()
                last_state = env.get_state()
                h_state = [None] * env.n_agent
                tgt_h_state = [None] * env.n_agent
                last_action = [-1] * env.n_agent
                step = 0
                R = 0.0

                child.send((last_obs, h_state, tgt_h_state, last_action))
            elif cmd == "close":
                child.close()
                break
            elif cmd == "get_rand_states":
                rand_states = {
                    "random_state": random.getstate(),
                    "np_random_state": np.random.get_state(),
                }
                child.send(rand_states)
            elif cmd == "load_rand_states":
                random.setstate(data["random_state"])
                np.random.set_state(data["np_random_state"])
            else:
                raise NotImplementerError

    except KeyboardInterrupt:
        print("EnvRunner worker: caught keyboard interrupt")
    except Exception as e:
        print("EnvRunner worker: uncaught worker exception")
        raise


class EnvsRunner(object):
    """
    Environment runner which runs multiple environemnts in parallel in subprocesses
    and communicates with them via pipe
    """

    def __init__(
        self,
        env,
        n_envs,
        controller,
        memory,
        max_epi_steps,
        gamma,
        seed,
        obs_last_action=False,
    ):
        """
        Parameters
        ----------
        env: gym.env
            a gym environment
        n_envs: int
            the number of parallel envs
        controller: MAC
            an instance of the MAC class (see ./controller.py)
        memory: Memory
            an instance of the Memmory class (see ./memory.py)
        max_epi_steps: int
            the maximum time steps of each episode
        gamma: float
            a discount factor
        seed: int
            a random seed
        obs_last_action: bool
            whether having the last action in observation or not
        """

        self.env = env
        self.max_epi_steps = max_epi_steps
        self.n_envs = n_envs
        self.n_agent = env.n_agent
        self.controller = controller
        self.parents, self.children = [list(i) for i in zip(*[Pipe() for _ in range(n_envs)])]
        self.envs = [
            Process(target=worker, args=(child, env, gamma, seed + idx))
            for idx, child in enumerate(self.children)
        ]
        self.memory = memory
        self.obs_last_action = obs_last_action
        # collect the episode data for each env
        self.episodes = [[] for i in range(n_envs)]
        # collect training returns 
        self.returns = []

        for env in self.envs:
            env.daemon = True
            env.start()

        for child in self.children:
            child.close()

    def run(self, eps, n_epis=1):
        """
        Rollout n episodes

        Parameters
        ----------
        eps: float
            the epsilon value for exploration
        n_epis: int
            the number of episodes to run
        """

        self._reset()
        while self.n_epi_count < n_epis:
            self._step(eps)

    def close(self):
        """
        Close all processes
        """

        [parent.send(("close", None)) for parent in self.parents]
        [parent.close() for parent in self.parents]
        [env.terminate() for env in self.envs]
        [env.join() for env in self.envs]

    def _step(self, eps):
        """
        All parallel envs run one step

        Parameter
        ---------
        eps: float
            the epsilon value for exploration
        """

        for idx, parent in enumerate(self.parents):

            # get actions for agents
            actions, self.h_states[idx] = self.controller.select_action(
                self.last_obses[idx], self.h_states[idx], eps=eps
            )
            # make target-net hidden state up-to-date
            # the target-net is used for choosing target-action for on-policy learning later
            _, self.tgt_h_states[idx] = self.controller.select_action(
                self.last_obses[idx], self.tgt_h_states[idx], eps=eps, using_tgt_net=True
            )
            # send actions to parallel envs and run one step
            parent.send(("step", actions))
            self.step_count[idx] += 1

        # collect envs' returns
        for idx, parent in enumerate(self.parents):
            env_return = parent.recv()
            env_return = self._exp_to_tensor(idx, env_return, eps)
            self.episodes[idx].append(env_return)
            self.last_obses[idx] = env_return[5]
            if self.obs_last_action:
                self.last_actions[idx] = env_return[2]

            # if episode is done, add it to memory buffer
            if env_return[-3][0] or self.step_count[idx] == self.max_epi_steps:
                self.n_epi_count += 1
                self.memory.scenario_cache += self.episodes[idx]
                self.memory.flush_buf_cache()
                # collect returns from all envs
                parent.send(("get_return", None))
                R = parent.recv()
                self.returns.append(R)
                # when episode is done, reset env
                parent.send(("reset", None))
                (
                    self.last_obses[idx],
                    self.h_states[idx],
                    self.tgt_h_states[idx],
                    self.last_actions[idx],
                ) = parent.recv()
                self.last_obses[idx] = self.obs_to_tensor(self.last_obses[idx])
                if self.obs_last_action:
                    self.last_actions[idx] = self.action_to_tensor(self.last_actions[idx])
                    self.last_obses[idx] = self.rebuild_obs(
                        self.env, self.last_obses[idx], self.last_actions[idx]
                    )
                self.episodes[idx] = []
                self.step_count[idx] = 0

    def _reset(self):
        """
        Reset all envs and variables
        """

        # send cmd to reset envs
        for parent in self.parents:
            parent.send(("reset", None))

        self.last_obses, self.h_states, self.tgt_h_states, self.last_actions = [
            list(i) for i in zip(*[parent.recv() for parent in self.parents])
        ]
        self.last_obses = [
            self.obs_to_tensor(obs) for obs in self.last_obses
        ]  # List[List[tensor]]
        if self.obs_last_action:
            self.last_actions = [self.action_to_tensor(a) for a in self.last_actions]
            # reconstruct obs to observe actions
            self.last_obses = [
                self.rebuild_obs(self.env, obs, a)
                for obs, a in zip(*[self.last_obses, self.last_actions])
            ]
        self.n_epi_count = 0
        self.step_count = [0] * self.n_envs
        self.episodes = [[] for i in range(self.n_envs)]

    def _exp_to_tensor(self, env_idx, exp, eps):
        """
        Make date to be Tensors

        Parameters
        ----------
        env_idx: int
            the ID of a env on a process
        exp: tuple
            a tupe of transition data at one time step (s, o, a, r, s', o', t, discount) 
        eps: float
            the epsilon value for exploration
        
        Return
        ------
        tuple(torch.FloatTensor)
        """

        last_state = torch.from_numpy(exp[0]).float().view(1, -1)
        last_obs = [torch.from_numpy(o).float().view(1, -1) for o in exp[1]]
        a = [torch.tensor(a).view(1, -1) for a in exp[2]]
        r = [torch.tensor(r).float().view(1, -1) for r in exp[3]]
        state = torch.from_numpy(exp[4]).float().view(1, -1)
        obs = [torch.from_numpy(o).float().view(1, -1) for o in exp[5]]
        # re-construct obs if observe last action
        if self.obs_last_action:
            last_obs = self.rebuild_obs(self.env, last_obs, self.last_actions[env_idx])
            obs = self.rebuild_obs(self.env, obs, a)
        # get a target action under new obs using the target actor net
        n_tgt_a, _ = self.controller.select_action(
            obs, self.tgt_h_states[env_idx], eps=eps, using_tgt_net=True
        )
        n_tgt_a = [torch.tensor(a).view(1, -1) for a in n_tgt_a]
        t = [torch.tensor(t).float().view(1, -1) for t in exp[6]]
        discount = [torch.tensor(exp[7]).float().view(1, -1)] * self.n_agent
        exp_v = [torch.tensor([1.0]).view(1, -1)] * self.n_agent

        return (last_state, last_obs, a, r, state, obs, n_tgt_a, t, discount, exp_v)

    @staticmethod
    def obs_to_tensor(obs):
        return [torch.from_numpy(o).float().view(1, -1) for o in obs]

    @staticmethod
    def action_to_tensor(action):
        return [torch.tensor(a).view(1, -1) for a in action]

    @staticmethod
    def rebuild_obs(env, obs, action):
        """
        Concatenate an observation and an action (one-hot vector) for each agent

        Parameters
        ----------
        obs: List[torch.FloatTensor]
            a list of agents' observations
        action: List[torch.FloatTensor]
            a list of agents' actions
        """

        new_obs = []
        for o, a, a_dim in zip(*[obs, action, env.n_action]):
            if a == -1:
                one_hot_a = torch.zeros(a_dim).view(1, -1)
            else:
                one_hot_a = F.one_hot(a.view(-1), a_dim).float()
            new_obs.append(torch.cat([o, one_hot_a], dim=1))
        return new_obs
