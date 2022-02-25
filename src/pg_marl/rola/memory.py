import numpy as np
import torch

from collections import deque


class Memory:
    """
    Base class of a memory buffer.

    """

    def __init__(self, env, obs_last_action=False, size=1):
        """
        Parameters
        ----------
        env: gym.env
            an instance of a gym environment
        obs_last_action: bool
            whether having the last action in observation or not
        size: int
            the size of the buffer
        """

        # create a buffer based on the requested size
        self.buf = deque(maxlen=size)
        # the number of agents in the env
        self.n_agent = env.n_agent
        # create transition paddings
        self.ZERO_STATE = torch.zeros(env.get_env_info()["state_shape"]).view(1, -1)
        if not obs_last_action:
            self.ZERO_OBS = [torch.zeros(dim).view(1, -1) for dim in env.obs_size]
        else:
            self.ZERO_OBS = [
                torch.zeros(o_dim + a_dim).view(1, -1)
                for o_dim, a_dim in zip(*[env.obs_size, env.n_action])
            ]
        self.ZERO_ACT = [torch.tensor(0).view(1, -1)] * self.n_agent
        self.ZERO_REWARD = [torch.tensor(0.0).view(1, -1)] * self.n_agent
        self.ONE_TERMINATE = [torch.tensor(1.0).view(1, -1)] * self.n_agent
        self.ZERO_DISCOUNT = [torch.tensor(0.0).view(1, -1)] * self.n_agent
        # indicate whether the current trainsition experience is valid or not to distinguish paddings and real trainsitions
        self.ZERO_EXPV = [torch.tensor(0.0).view(1, -1)] * self.n_agent

        self.ZERO_PADDING = [
            (
                self.ZERO_STATE,
                self.ZERO_OBS,
                self.ZERO_ACT,
                self.ZERO_REWARD,
                self.ZERO_STATE,
                self.ZERO_OBS,
                self.ZERO_ACT,
                self.ONE_TERMINATE,
                self.ZERO_DISCOUNT,
                self.ZERO_EXPV,
            )
        ]

    def append(self, transition):
        """
        Append a new transition
        """

        raise NotImplementedError

    def flush_buf_cache(self):
        """
        Add the latest episodes into buffer and reset cache

        """

        raise NotImplementedError

    def sample(self):
        """
        Sample sequences from the buffer
        """

        raise NotImplementedError

    def _scenario_cache_reset(self):
        """
        Reset cache
        """

        raise NotImplementedError


class MemoryEpi(Memory):
    """
    A child class of Memory
    """

    def __init__(self, *args, **kwargs):
        super(MemoryEpi, self).__init__(*args, **kwargs)
        self._scenario_cache_reset()

    def flush_buf_cache(self):
        self.buf.append(self.scenario_cache)
        self._scenario_cache_reset()

    def sample(self):
        batch = list(self.buf)
        return self._padding_batches(batch)

    def append(self, transition):
        """
        Parameter
        ---------
        transition: tuple
            a tuple include the latest transition data
        """
        self.scenario_cache.append(transition)

    def _scenario_cache_reset(self):
        self.scenario_cache = []

    def _padding_batches(self, batch):
        """
        Padding each episodes to make them have a same length

        Parameter
        ---------
        batch: List[List[tuple]]
            a list of episodes, each episode is a list of trainsition tuples.

        Returns
        -------
        List[List[tuple]]
            a list of padded episodes
        int
            the maximum original episode length in the sampled batch
        List[int]
            a list of original length of each sampled episode
        """

        epi_lens = [len(epi) for epi in batch]
        max_len = max(epi_lens)
        batch = [epi + self.ZERO_PADDING * (max_len - len(epi)) for epi in batch]
        return batch, max_len, epi_lens
