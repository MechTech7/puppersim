from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gin
from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils

@gin.configurable
class SimpleAgilityTask(task_interface.Task):
    def __init__(self, terminal_condition=terminal_conditions):
        self._step_count = 0
        self._terminal_condition = terminal_condition
        self._env = None

        self.des_dir = np.array([0, 0], dtype=np.float32)
        self.des_velocity = 0.0

        self._divide_with_dt = True

        self._last_base_position = None

        self.epsilon = 0.0001
        

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        self._env = env
        self._last_base_position = env_utils.get_robot_base_position(
            self._env.robot)

    @property
    def set_count(self):
        return self._step_count

    def update(self, env):
        """Updates the internal state of the task."""
        del env
        self._last_base_position = env_utils.get_robot_base_position(
            self._env.robot)

    def reward(self, env):
        del env

        self._step_count += 1
        env = self._env

        current_base_position = env_utils.get_robot_base_position(self._env.robot)
        velocity = current_base_position - self._last_base_position

        if self._divide_with_dt:
            velocity /= env.env_time_step

        vel_norm = np.linalg.norm(velocity)
        normed_velocity = velocity / vel_norm

        vector_rew = np.dot(normed_velocity, self.des_velocity)
        speed_rew = 1 / (vel_norm - self.des_velocity + self.epsilon)

        reward = vector_rew + speed_rew

        return reward
    
    def done(self, env):
        del env

        position = env_utils.get_robot_base_position(self._env.robot)
        if self._min_com_height and position[2] < self._min_com_height:
            return True
        
        return self._terminal_conditions(self._env)

