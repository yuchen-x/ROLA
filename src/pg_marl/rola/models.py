import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def Linear(input_dim, output_dim, act_fn="leaky_relu", init_weight_uniform=True):
    """
    Create a fully-connected layer and initialize weights

    Parameters:
    ----------
    input_dim: int
        the input dimension
    output_dim: int
        the output dimension
    act_fn: str
        the activation function
    init_weight_uniform: bool
        whether sampling weigths using a uniform distribution

    Return
    ------
    torch.nn.Linear
        an initilized fully-connected layer
    """

    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


class Actor(nn.Module):
    """
    An actor model represents a policy network
    """

    def __init__(self, input_dim, output_dim, mlp_layer_size=64, rnn_layer_size=64):
        """
        Parameters
        ----------
        input_dim: int
            the input dimension
        output_dim: int
            the ouput dimension
        mlp_layer_size: int
            the number of neurons in each fully-connected layer
        rnn_layer_size: int
            the number of neurons in each rnn layer
        """
        super(Actor, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size, act_fn="leaky_relu")
        self.lstm = nn.LSTM(
            rnn_layer_size, hidden_size=rnn_layer_size, num_layers=1, batch_first=True
        )
        self.fc2 = Linear(mlp_layer_size, output_dim, act_fn="linear")

    def forward(self, x, h=None, eps=0.0, test_mode=False):
        """
        Forward passing through the network

        Parameters
        ----------
        x: torch.FloatTensor
            input tensor
        h: torch.FloatTensor or None
            the hidden state maintained by the RNN layer
        eps: float
            epsilon's value for exploration during training
        test_mode: false
            whether using test_mode or not

        Return
        ------
        torch.FloatTensor
            the logits of actions
        torch.FloatTensor
            the new hidden state maintained by the RNN layer
        """

        x = F.leaky_relu(self.fc1(x))
        x, h = self.lstm(x, h)
        x = self.fc2(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + np.log(1 - eps)
            logits_2 = torch.full_like(
                action_logits, np.log(eps) - np.log(action_logits.size(-1))
            )
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits, axis=0)

        return action_logits, h


class Critic(nn.Module):
    """
    An critic model represents an action-value function
    """

    def __init__(self, input_dim, output_dim=1, mlp_layer_size=64):
        """
        Parameters
        ----------
        input_dim: int
            the input dimension
        output_dim: int
            the ouput dimension
        mlp_layer_size: int
        """
 
        super(Critic, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size, act_fn="leaky_relu")
        self.fc2 = Linear(mlp_layer_size, mlp_layer_size, act_fn="leaky_relu")
        self.fc3 = Linear(mlp_layer_size, output_dim, act_fn="linear")

    def forward(self, x):
        """
        Forward passing through the network

        Parameters
        ----------
        x: torch.FloatTensor
            input tensor

        Return
        ------
        torch.FloatTensor
            the values of actions
        """

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        Q_value = self.fc3(x)

        return Q_value
