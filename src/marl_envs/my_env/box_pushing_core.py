#!/usr/bin/python

import numpy as np

DIRECTION = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])


class Agent(object):
    """
    An agent class for contolling the agent as well as maintaining attributes
    """

    def __init__(self, idx, init_x, init_y, init_ori, grid_dim):

        """
        Parameters
        ----------
        idx: int
            agent's ID
        init_x: float
            initial x-coordinate
        init_y: float
            initial y-coordinate
        init_ori: int
            initial orientation (discrete space)
        gird_dim: tupe(int,int)
            the size of a grid world
        """

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0, 0.0])
        self.cur_action = None
        self.xlen, self.ylen = grid_dim

    def step(self, action, boxes):

        """
        Parameters
        ----------
        action: int
            an action's index
        boxes: List[Box]
            a list of boxes in the env

        Return
        ------
        float:
            a reward value
        """

        assert action < 4, "The action received is out of range"

        reward = 0.0

        self.cur_action = action

        # action: move forward
        if action == 0:
            # moving direction depends on agent's current orientation
            move = DIRECTION[self.ori]
            self.xcoord += move[0]
            self.ycoord += move[1]
            # check if touch env's boundaries
            if (
                self.xcoord > self.xlen - 0.5
                or self.xcoord < 0.5
                or self.ycoord > self.ylen - 0.5
                or self.ycoord < 0.5
            ):
                self.xcoord -= move[0]
                self.ycoord -= move[1]
                reward += -5.0
            # check if push small box
            for box in boxes:
                if (
                    box.xcoord == self.xcoord
                    or box.xcoord == self.xcoord - 0.5
                    or box.xcoord == self.xcoord + 0.5
                ) and box.ycoord == self.ycoord:
                    if self.ori == 0 and box.idx != 2:
                        box.xcoord += move[0]
                        box.ycoord += move[1]
                    else:
                        self.xcoord -= move[0]
                        self.ycoord -= move[1]
                        reward += -5.0
        # turn left
        elif action == 1:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1
        # turn right
        elif action == 2:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1

        return reward


class Box(object):

    """A box class with attributes"""

    def __init__(self, idx, init_x, init_y, size_h, size_w):

        """
        Parameters
        ----------
        idx: int
            the box's ID
        init_x: float
            initial x-coordinate
        init_y: float
            initial y-coordinate
        size_h: float
            the height of the box
        size_w: float
            the width of the box
        """

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.h = size_h
        self.w = size_w
