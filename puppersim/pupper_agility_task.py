from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pybullet_envs.minitaur.envs_v2.sensors import sensor

import gin
from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils

@gin.configurable
class SimpleAgilityTask(task_interface.Task):
    def __init__(self, 
                terminal_condition=terminal_conditions,
                min_com_height=None,
                sensor_name="desired_direction", 
                alpha=0.5, 
                dev_norm=0.01, 
                k_steps=2000):

        self._step_count = 0
        self._terminal_conditions = terminal_condition
        self._env = None

        self.sensor_name = sensor_name
        #Objective Variables
        self.des_velocity = np.array([0, 0], dtype=np.float32)
        self.des_speed = 0.0
        self.normed_des = np.array([0, 0], dtype=np.float32)

        self._alpha = alpha #relative importance of speed and direction
        self._deviation_norm = dev_norm
        self.k_steps = k_steps


        self._min_com_height = min_com_height
        self._divide_with_dt = True

        self._last_base_position = None

        self._epsilon = 0.0001
        

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        self._env = env
        self._last_base_position = np.array(env_utils.get_robot_base_position(
            self._env.robot))

        self.des_velocity = np.zeros(2, dtype=np.float32)
        self.normed_des = np.zeros(2, dtype=np.float32)
        self.des_speed = 0.0

    @property
    def set_count(self):
        return self._step_count

    def update(self, env):
        """Updates the internal state of the task."""
        del env
        self._last_base_position = np.array(env_utils.get_robot_base_position(
            self._env.robot))

    def velocity_update(self):
        #Every k timesteps, perturb the desired velocity vector in 
        random_vec = np.random.random(size=2)
        vec_norm = np.linalg.norm(random_vec)
        if vec_norm != 0:
            random_vec = random_vec / np.linalg.norm(random_vec)
        pass

        random_vec = self._deviation_norm * random_vec

        self.des_velocity = self.des_velocity + random_vec
        self.des_speed = np.linalg.norm(self.des_velocity)
        self.normed_des = self.des_velocity / self.des_speed

    def reward(self, env):
        del env

        self._step_count += 1
        env = self._env

        #TODO: Begin Testing Here
        obs_dict = env._observation_dict[self.sensor_name]

        current_base_position = np.array(env_utils.get_robot_base_position(self._env.robot))
        velocity = current_base_position - self._last_base_position
        velocity = velocity[:2]

        if self._divide_with_dt:
            velocity /= env.env_time_step

        vel_speed = np.linalg.norm(velocity)
        normed_velocity = velocity / vel_speed

        direction_rew = np.dot(normed_velocity, self.normed_des)
        speed_rew = 1 / (vel_speed - self.des_speed + self._epsilon)


        if self._step_count % self.k_steps:
            self.velocity_update()
        
        reward = self._alpha * direction_rew + (1 - self._alpha) * speed_rew

        """
        print("-----------------")
        print(f"des_vel: {self.des_velocity}")
        print(f"reward: {reward}")
        print("-----------------")
        """
        
        return reward
    
    def done(self, env):
        del env

        position = env_utils.get_robot_base_position(self._env.robot)
        if self._min_com_height and position[2] < self._min_com_height:
            return True
        
        return self._terminal_conditions(self._env)

