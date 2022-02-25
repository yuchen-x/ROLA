import torch
import copy
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.distributions import Categorical
from itertools import chain
from .models import Critic


class Learner(object):

    """
    A class of ROLA learner

    """

    def __init__(
        self,
        env,
        env_name,
        controller,
        memory,
        gamma=0.95,
        a_lr=1e-2,
        c_lr=1e-2,
        local_c_train_iteration=1,
        c_target_update_freq=50,
        cen_c_mlp_layer_size=64,
        local_c_mlp_layer_size=64,
        tau=0.01,
        grad_clip_value=None,
        grad_clip_norm=None,
        MC_baseline=False,
        n_step_bootstrap=0,
    ):

        """
        Parameters
        ----------
        env: gym.env
            a gym environment
        env_name: str
            the name of a domain
        controller: MAC
            an instance of the MAC class (see ./controller.py)
        memory: Memory
            an instance of the Memmory class (see ./memory.py)
        gamma: float
            a discount factor
        a_lr: float
            the learning rate for actor
        c_lr: float
            the learning rate for cirtic
        local_c_train_iteration: int
            the number of iteration to train local critic using the centralized critic to sample target actions
        c_target_update_freq: int
            the frequency to update the target-net of critic
        cen_c_mlp_layer_size: int
            the number of neurons on the fully-connected layers of the centralized critic
        local_c_mlp_layer_size: int
            the number of neurons on the fully-connected layers of the local critic
        tau: float
            the updating rate for target-net soft updates
        grad_clip_value: float
            gradient clipping value
        grad_clip_norm: None/float
            gradient clipping norm
        MC_baseline: bool
            whether use Reinforce with baseline or not 
        n_step_bootstrap: int
            n-step TD
        """

        self.env = env
        self.env_name = env_name
        self.n_agent = env.n_agent
        self.agents_action_space = env.n_action
        self.controller = controller
        self.memory = memory
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.local_c_train_iteration = local_c_train_iteration
        self.cen_c_mlp_layer_size = cen_c_mlp_layer_size
        self.local_c_mlp_layer_size = local_c_mlp_layer_size
        self.c_target_update_freq = c_target_update_freq
        self.tau = tau
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.MC_baseline = MC_baseline
        self.n_step_bootstrap = n_step_bootstrap

        self._create_cen_critic()
        self._create_local_critic()
        self._set_optimizer()

    def train(self, eps):
        """
        Training for critics and actors

        Parameters:
        eps: float
            the current epsilon value for exploration
        """

        # request on-policy training data
        batch, trace_len, epi_len = self.memory.sample()
        batch_size = len(batch)

        #################################################################################
        #################### Step-1: train a centralized critic #########################
        #################################################################################

        # prepare centralized training data
        cen_batch = self._cat_joint_exps(batch)
        state, action, reward, n_state, n_tgt_action, terminate, discnt, exp_v = zip(*cen_batch)
        # shape training data
        cen_state = torch.cat(state).view(batch_size, trace_len, -1)
        cen_action = torch.cat(action).view(batch_size, trace_len, -1)
        cen_reward = torch.cat(reward).view(batch_size, trace_len, -1)
        cen_n_state = torch.cat(n_state).view(batch_size, trace_len, -1)
        cen_n_tgt_action = torch.cat(n_tgt_action).view(batch_size, trace_len, -1)
        cen_terminate = torch.cat(terminate).view(batch_size, trace_len, -1)
        cen_discnt = torch.cat(discnt).view(batch_size, trace_len, -1)
        cen_exp_v = torch.cat(exp_v).view(batch_size, trace_len, -1)

        if not self.MC_baseline:
            # get n-step return
            Gt = self._get_bootstrap_return(
                cen_reward,
                cen_n_state,
                cen_n_tgt_action,
                cen_terminate,
                epi_len,
                self.cen_critic_tgt_net,
            )
        else:
            # get monte carlo return
            Gt = self._get_discounted_return(
                cen_reward,
                cen_n_state,
                cen_n_tgt_action,
                cen_terminate,
                epi_len,
                self.cen_critic_tgt_net,
            )

        # compute TD loss and optimize the centtralized critic
        TD = Gt - self.cen_critic_net(cen_state).gather(-1, cen_action)
        self.cen_critic_loss = torch.sum(cen_exp_v * TD * TD) / cen_exp_v.sum()
        self.cen_critic_optimizer.zero_grad()
        self.cen_critic_loss.backward()
        if self.grad_clip_value is not None:
            clip_grad_value_(self.cen_critic_net.parameters(), self.grad_clip_value)
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.cen_critic_net.parameters(), self.grad_clip_norm)
        self.cen_critic_optimizer.step()

        #################################################################################
        ######### Step-2: train a local critic for each agent using the ################# 
        ######### centralized one to sample target actions ##############################
        #################################################################################

        # get the q-values at the next states of each trainsition tuple
        cen_critic_Q_next = self.cen_critic_net(cen_n_state).detach()
        for _ in range(self.local_c_train_iteration):
            # sample joint actions at next states
            cen_action_next = (
                Categorical(logits=F.log_softmax(cen_critic_Q_next, dim=-1))
                .sample()
                .unsqueeze(-1)
            )
            # translate from the index of a joint action to the index of each agent's individual action
            id_action_next = tuple(
                [
                    torch.from_numpy(a)
                    for a in np.unravel_index(cen_action_next, self.agents_action_space)
                ]
            )

            # prepare decentralized training data 
            dec_batch = self._sep_joint_exps(batch)

            for agent, ag_batch in zip(self.controller.agents, dec_batch):

                state, obs, action, reward, n_state, terminate, discnt, exp_v = zip(*ag_batch)

                # shape training data
                state = torch.cat(state).view(batch_size, trace_len, -1)
                action = torch.cat(action).view(batch_size, trace_len, -1)
                reward = torch.cat(reward).view(batch_size, trace_len, -1)
                n_state = torch.cat(n_state).view(batch_size, trace_len, -1)
                terminate = torch.cat(terminate).view(batch_size, trace_len, -1)
                discnt = torch.cat(discnt).view(batch_size, trace_len, -1)
                exp_v = torch.cat(exp_v).view(batch_size, trace_len, -1)

                if not self.MC_baseline:
                    # get n-step return
                    Gt = self._get_bootstrap_return(
                        reward,
                        n_state,
                        id_action_next[agent.idx],
                        terminate,
                        epi_len,
                        agent.critic_tgt_net,
                    )
                else:
                    # get monte carlo return
                    Gt = self._get_discounted_return(
                        reward,
                        n_state,
                        id_action_next[agent.idx],
                        terminate,
                        epi_len,
                        agent.critic_tgt_net,
                    )

                # compute TD loss 
                TD = Gt - agent.critic_net(state).gather(-1, action)
                agent.critic_loss = torch.sum(exp_v * TD * TD) / exp_v.sum()

            # optimize each agent's local  critic
            for agent in self.controller.agents:
                agent.critic_optimizer.zero_grad()
                agent.critic_loss.backward()
                if self.grad_clip_value is not None:
                    clip_grad_value_(agent.critic_net.parameters(), self.grad_clip_value)
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(agent.critic_net.parameters(), self.grad_clip_norm)
                agent.critic_optimizer.step()

        #################################################################################
        ################### Step-3: update each agent's actor-net #######################
        #################################################################################

        for agent, batch in zip(self.controller.agents, dec_batch):

            state, obs, action, reward, n_state, terminate, discnt, exp_v = zip(*batch)

            state = torch.cat(state).view(batch_size, trace_len, -1)
            obs = torch.cat(obs).view(batch_size, trace_len, -1)
            action = torch.cat(action).view(batch_size, trace_len, -1)
            discnt = torch.cat(discnt).view(batch_size, trace_len, -1)
            exp_v = torch.cat(exp_v).view(batch_size, trace_len, -1)

            # compute a basline using each agent's policy and local critic
            local_baseline = torch.sum(
                Categorical(logits=agent.actor_net(obs, eps=eps)[0].detach()).probs
                * agent.critic_net(state).detach(),
                dim=-1,
                keepdim=True,
            )
            # compute an advantage value estimation
            local_adv_value = (
                agent.critic_net(state).detach().gather(-1, action) - local_baseline
            )
            # compute policy gradient with a baseline
            action_logits = agent.actor_net(obs, eps=eps)[0]
            log_pi_a = action_logits.gather(-1, action)
            actor_loss = torch.sum(exp_v * discnt * (log_pi_a * local_adv_value), dim=1)
            agent.actor_loss = -1 * torch.sum(actor_loss) / exp_v.sum()

        # optimize each actor-net
        for agent in self.controller.agents:
            agent.actor_optimizer.zero_grad()
            agent.actor_loss.backward()
            if self.grad_clip_value is not None:
                clip_grad_value_(agent.actor_net.parameters(), self.grad_clip_value)
            if self.grad_clip_norm is not None:
                clip_grad_norm_(agent.actor_net.parameters(), self.grad_clip_norm)
            agent.actor_optimizer.step()

    def update_critic_target_net(self, soft=False):
        """
        Update the parameters of critic's target-net
        """

        if not soft:
            self.cen_critic_tgt_net.load_state_dict(self.cen_critic_net.state_dict())
            for agent in self.controller.agents:
                agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())
        else:
            with torch.no_grad():
                for q, q_targ in zip(
                    self.cen_critic_net.parameters(), self.cen_critic_tgt_net.parameters()
                ):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)
            for agent in self.controller.agents:
                for q, q_targ in zip(
                    agent.critic_net.parameters(), agent.critic_tgt_net.parameters()
                ):
                    q_targ.data.mul_(1 - self.tau)
                    q_targ.data.add_(self.tau * q.data)

    def update_actor_target_net(self, soft=False):
        """
        Update the parameters of actor's target-net
        """

        for agent in self.controller.agents:
            if not soft:
                agent.actor_tgt_net.load_state_dict(agent.actor_net.state_dict())
            else:
                with torch.no_grad():
                    for q, q_targ in zip(
                        agent.actor_net.parameters(), agent.actor_tgt_net.parameters()
                    ):
                        q_targ.data.mul_(1 - self.tau)
                        q_targ.data.add_(self.tau * q.data)

    def _create_cen_critic(self):
        """
        Create a centralized critic model
        """

        input_dim = self._get_critic_input_shape()
        output_dim = self._get_cen_output_shape()
        self.cen_critic_net = Critic(input_dim, output_dim, self.cen_c_mlp_layer_size)
        self.cen_critic_tgt_net = Critic(input_dim, output_dim, self.cen_c_mlp_layer_size)
        self.cen_critic_tgt_net.load_state_dict(self.cen_critic_net.state_dict())

    def _create_local_critic(self):
        """
        Create a local critic model for each agent
        """

        input_dim = self._get_critic_input_shape()
        for agent in self.controller.agents:
            agent.critic_net = Critic(
                input_dim, self.env.n_action[agent.idx], self.local_c_mlp_layer_size
            )
            agent.critic_tgt_net = Critic(
                input_dim, self.env.n_action[agent.idx], self.local_c_mlp_layer_size
            )
            agent.critic_tgt_net.load_state_dict(agent.critic_net.state_dict())

    def _get_critic_input_shape(self):
        """
        Get the input dimension of critic-net
        """

        return self.env.get_env_info()["state_shape"]

    def _get_cen_output_shape(self):
        """
        Get the output dimension of the centralized critic-net
        """

        return np.prod(self.env.n_action)

    def _set_optimizer(self):
        """
        Set optimizers for actors and critics
        """

        for agent in self.controller.agents:
            agent.actor_optimizer = Adam(agent.actor_net.parameters(), lr=self.a_lr)
            agent.critic_optimizer = Adam(agent.critic_net.parameters(), lr=self.c_lr)
        self.cen_critic_optimizer = Adam(self.cen_critic_net.parameters(), lr=self.c_lr)

    def _cat_joint_exps(self, joint_exps):
        """
        Process the sampled batch of joint experiences for training a centralized critic, and 
        the centralized critic only receives ground truth state as input

        Parameters
        ----------
        joint_exps: List[List[tuple(torch.FloatTensor)]]
            a list of episodes composed of trainsition tuples
        """

        exp = []
        if self.env_name != "pomdp_advanced_spread":
            for s, o, a, r, s_n, o_n, tgt_a, t, discnt, exp_v in chain(*joint_exps):
                exp.append(
                    [
                        s,
                        torch.tensor(np.ravel_multi_index(a, self.agents_action_space)).view(
                            1, -1
                        ),
                        r[0],
                        s_n,
                        torch.from_numpy(np.ravel_multi_index(tgt_a, self.agents_action_space)),
                        t[0],
                        discnt[0],
                        exp_v[0],
                    ]
                )
        else:
            # pomdp_advanced_spread having individual reward function for each agent rather than a 
            # global one, so that we sum each agent's reward as a global reward at each time step
            for s, o, a, r, s_n, o_n, tgt_a, t, discnt, exp_v in chain(*joint_exps):
                exp.append(
                    [
                        s,
                        torch.tensor(np.ravel_multi_index(a, self.agents_action_space)).view(
                            1, -1
                        ),
                        sum(r),
                        s_n,
                        torch.from_numpy(np.ravel_multi_index(tgt_a, self.agents_action_space)),
                        t[0],
                        discnt[0],
                        exp_v[0],
                    ]
                )

        return exp

    def _sep_joint_exps(self, joint_exps):
        """
        Seperate joint experiences for individual agent in order to train local critic and actor

        Parameters
        ----------
        joint_exps: List[List[tuple(torch.FloatTensor)]]
            a list of episodes composed of trainsition tuples
        """

        exps = [[] for _ in range(self.n_agent)]
        for s, o, a, r, s_n, o_n, tgt_a, t, discnt, exp_v in chain(*joint_exps):
            for i in range(self.n_agent):
                exps[i].append([s, o[i], a[i], r[i], s_n, t[i], discnt[i], exp_v[i]])
        return exps

    def _get_discounted_return(
        self, reward, n_state, n_tgt_action, terminate, epi_len, critic_net
    ):
        """
        Compute monte carlo return for each time step

        Parameters:
        ----------
        reward: torch.FloatTensor
            a tensor of rewards
        n_state: torch.FloatTensor
            a tensor of next states
        n_tgt_action: torch.FloatTensor
            a tensor of target actions at next states
        terminate: torch.FloatTensor
            a tensor of binary value indicates whether the current step is a terminal step or not
        epi_len: List[int]
            the number of time steps of each episodes
        critic_net: Critic
            an instance of the critic model (see ./models.py)

        Return
        ------
        torch.FloatTensor
            a tensor of returns
        """

        Gt = copy.deepcopy(reward)
        for epi_idx, epi_r in enumerate(Gt):
            end_step_idx = epi_len[epi_idx] - 1
            if not terminate[epi_idx][end_step_idx]:
                epi_r[end_step_idx] += (
                    self.gamma
                    * critic_net(n_state[epi_idx][end_step_idx]).detach()[
                        n_tgt_action[epi_idx][end_step_idx]
                    ]
                )
            for idx in range(end_step_idx - 1, -1, -1):
                epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx + 1]
        return Gt

    def _get_bootstrap_return(
        self, reward, n_state, n_tgt_action, terminate, epi_len, critic_net
    ):
        """
        Compute n-step return for each time step

        Parameters:
        ----------
        reward: torch.FloatTensor
            a tensor of rewards
        n_state: torch.FloatTensor
            a tensor of next states
        n_tgt_action: torch.FloatTensor
            a tensor of target actions at next states
        terminate: torch.FloatTensor
            a tensor of binary value indicates whether the current step is a terminal step or not
        epi_len: List[int]
            the number of time steps of each episodes
        critic_net: Critic
            an instance of the critic model (see ./models.py)

        Return
        ------
        torch.FloatTensor
            a tensor of returns
        """

        if self.n_step_bootstrap:
            # implement n-step bootstrap
            bootstrap = critic_net(n_state).detach().gather(-1, n_tgt_action)
            Gt = copy.deepcopy(reward)
            for epi_idx, epi_r in enumerate(Gt):
                end_step_idx = epi_len[epi_idx] - 1
                if not terminate[epi_idx][end_step_idx]:
                    epi_r[end_step_idx] += self.gamma * bootstrap[epi_idx][end_step_idx]
                for idx in range(end_step_idx - 1, -1, -1):
                    if idx > end_step_idx - self.n_step_bootstrap:
                        epi_r[idx] = epi_r[idx] + self.gamma * epi_r[idx + 1]
                    else:
                        epi_r[idx] = self._get_n_step_discounted_bootstrap_return(
                            reward[epi_idx][idx : idx + self.n_step_bootstrap],
                            bootstrap[epi_idx][idx + self.n_step_bootstrap - 1],
                        )
        else:
            Gt = reward + self.gamma * critic_net(n_state).detach().gather(-1, n_tgt_action) * (
                -terminate + 1
            )
        return Gt

    def _get_n_step_discounted_bootstrap_return(self, reward, bootstrap):
        """
        Comput n-step return for a particular time step

        """

        discount = torch.pow(
            torch.ones(1, 1) * self.gamma, torch.arange(self.n_step_bootstrap)
        ).view(self.n_step_bootstrap, 1)
        Gt = torch.sum(discount * reward) + self.gamma**self.n_step_bootstrap * bootstrap
        return Gt
