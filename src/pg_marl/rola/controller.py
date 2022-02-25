import torch

from torch.distributions import Categorical
from .envs_runner import EnvsRunner
from .models import Actor
from .utils import Agent


class MAC(object):
    """
    A class for controlling agents' using their policies, and also 
    have an evaluation function to test the current learned policies.

    """

    def __init__(self, env, obs_last_action=False, a_mlp_layer_size=64, a_rnn_layer_size=64):
        """
        Parameters
        ----------
        env: gym.env
            an instance of a gym environment
        obs_last_action: bool
            whether having the last action in observation or not
        a_mlp_layer_size: int
            the number of neurons in mlp layers of the actor-net
        a_rnn_layer_size: int
            the number of neurons in rnn layers of the actor-net
        """

        self.env = env
        self.n_agent = env.n_agent
        self.obs_last_action = obs_last_action
        self.a_mlp_layer_size = a_mlp_layer_size
        self.a_rnn_layer_size = a_rnn_layer_size
        # record evaluation returns
        self.eval_returns = []
        # create agents
        self._build_agents()

    def select_action(self, obses, h_states, eps=0.0, test_mode=False, using_tgt_net=False):
        """
        Select an action for each agent using the current policy

        Parameters
        ----------
        obses: List[torch.FloatTensor]
            a list of agents' observations
        h_states: List[torch.FloatTensor]
            a list of hidden states maintained by the rnn layer of each agent's policy-net
        eps: float
            epsilon's value for exploration during training
        test_mode: false
            whether using test_mode or not
        using_tgt_net: fool
            whether using the target policy-net (with delayed weights) or not

        Returns
        -------
        List[int]
            a list of indices of agents' actions
        List[torch.FloatTensor]
            a list of hidden states maintained by the rnn layer of each agent's policy-net
        """
        actions = []
        new_h_states = []
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                if not using_tgt_net:
                    action_logits, new_h_state = agent.actor_net(
                        obses[idx].view(1, 1, -1), h_states[idx], eps=eps, test_mode=test_mode
                    )
                else:
                    action_logits, new_h_state = agent.actor_tgt_net(
                        obses[idx].view(1, 1, -1), h_states[idx], eps=eps, test_mode=test_mode
                    )
                action_prob = Categorical(logits=action_logits[0])
                action = action_prob.sample().item()
                actions.append(action)
                new_h_states.append(new_h_state)
        return actions, new_h_states

    def evaluate(self, gamma, max_epi_steps, n_episode=10):
        R = 0.0

        for _ in range(n_episode):
            t = 0
            step = 0
            last_obs = EnvsRunner.obs_to_tensor(self.env.reset())
            if self.obs_last_action:
                last_action = EnvsRunner.action_to_tensor([-1] * self.env.n_agent)
                last_obs = EnvsRunner.rebuild_obs(self.env, last_obs, last_action)
            h_state = [None] * self.n_agent

            while not t and step < max_epi_steps:

                a, h_state = self.select_action(last_obs, h_state, test_mode=True)
                obs, r, t, _ = self.env.step(a)
                last_obs = EnvsRunner.obs_to_tensor(obs)
                if self.obs_last_action:
                    a = EnvsRunner.action_to_tensor(a)
                    last_obs = EnvsRunner.rebuild_obs(self.env, last_obs, a)
                R += gamma**step * sum(r) / self.n_agent
                step += 1
                t = all(t)

        self.eval_returns.append(R / n_episode)

    def _build_agents(self):
        """
        Create a set of agents
        """

        self.agents = []
        for idx in range(self.n_agent):
            agent = Agent()
            agent.idx = idx
            agent.actor_net = Actor(
                self._get_input_shape(idx),
                self.env.n_action[idx],
                self.a_mlp_layer_size,
                self.a_rnn_layer_size,
            )
            agent.actor_tgt_net = Actor(
                self._get_input_shape(idx),
                self.env.n_action[idx],
                self.a_mlp_layer_size,
                self.a_rnn_layer_size,
            )
            agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            self.agents.append(agent)

    def _get_input_shape(self, agent_idx):
        """
        Get the input dimention of agent idx's policy-net

        Parameter
        ---------
        agent_idx: int
            the agent's ID

        Return
        ------
        int
            the agent's observation dimention or observation-action dimention 
        """

        if not self.obs_last_action:
            return self.env.obs_size[agent_idx]
        else:
            return self.env.obs_size[agent_idx] + self.env.n_action[agent_idx]
