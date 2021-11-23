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
                agility_sensor=None, 
                alpha=0.5, 
                dev_norm=0.01, 
                k_steps=2000):

        self._step_count = 0
        self._terminal_conditions = terminal_condition
        self._env = None

        self.sensor_name = sensor_name
        #Objective Variables
        self._agility_sensor = agility_sensor
        

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

        #self._agility_sensor.set_robot(self._env.robot)
        self._agility_sensor.update()

        self._last_base_position = np.array(env_utils.get_robot_base_position(
            self._env.robot))
        print(f"last_base_pos: {self._last_base_position}")

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
        
        self._agility_sensor.update()
        #print("-----------------")
        #print(f"task: {self._last_base_position}")
        #print(f"sensor: {self._agility_sensor._last_base_position}")

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

        des_vel = self._agility_sensor.get_des()
        velocity = self._agility_sensor.get_vel()

        des_speed = np.linalg.norm(des_vel)
        normed_des = des_vel

        if des_speed != 0:
            normed_des /= des_speed

        
        if self._divide_with_dt:
            velocity /= env.env_time_step
        

        vel_speed = np.linalg.norm(velocity)
        normed_velocity = velocity / vel_speed

        direction_rew = np.dot(normed_velocity, normed_des)
        speed_rew = 1 / (vel_speed - des_speed + self._epsilon)
        
        #TODO: Change the Reward
        reward = self._alpha * direction_rew + (1 - self._alpha) * speed_rew

        
        """
        print("-----------------")
        print(f"des_vel: {des_vel}")
        print(f"robot_vel: {velocity}")
        print(f"reward: {reward}")
        """
    
        return reward
    
    def done(self, env):
        del env

        position = env_utils.get_robot_base_position(self._env.robot)
        if self._min_com_height and position[2] < self._min_com_height:
            return True
        
        return self._terminal_conditions(self._env)

    @property
    def sensors(self):
        return [self._agility_sensor]

