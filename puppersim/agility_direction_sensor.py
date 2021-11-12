import numpy as np
import gin
import gym

from pybullet_envs.minitaur.envs_v2.sensors import sensor
from pybullet_envs.minitaur.envs_v2.utilities import noise_generators

@gin.configurable
class DesiredDirectionSensor(sensor.Sensor):
    def __init__(self, 
                k_steps=2000,
                name="desired_direction",
                sensor_latency=0.0):
        super().__init__(
            name=name,
            sensor_latency=sensor_latency,
            interpolator_fn=sensor.linear_obs_blender)

        self.des_velocity = np.array([0, 0], dtype=np.float32)
        self._k_steps = k_steps
        self._step_count = 0

    def _reset_des_vel(self):
        self.des_velocity = np.zeros(2, dtype=np.float32)
        self.normed_des = np.zeros(2, dtype=np.float32)
        self.des_speed = 0.0

    def on_reset(self, env):
        self._env = env
        self._observation_buffer.reset()
        self._reset_des_vel()
        self.on_new_observation()

    def velocity_update(self):
        #Every k timesteps, perturb the desired velocity vector in 
        random_vec = np.random.random(size=2)
        vec_norm = np.linalg.norm(random_vec)
        if vec_norm != 0:
            random_vec = random_vec / np.linalg.norm(random_vec)
        pass

        random_vec = self._deviation_norm * random_vec
        self.des_velocity = self.des_velocity + random_vec
    
    def _get_original_observation(self):
        ret_vel = np.copy(self.des_velocity)
        if self._step_count % self.k_steps:
            self.velocity_update()

        return self._robot.timestamp, ret_vel

    