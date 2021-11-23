import numpy as np
import gin
import gym
from numpy.random.mtrand import rand

from pybullet_envs.minitaur.envs_v2.sensors import sensor
from pybullet_envs.minitaur.envs_v2.utilities import noise_generators

@gin.configurable
class DesiredDirectionSensor(sensor.Sensor):
    def __init__(self, 
                name="desired_direction",
                k_steps=2000,
                dev_norm=0.1,
                lower_bound=np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float32),
                upper_bound=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
                sensor_latency=0.0):
        super().__init__(
            name=name,
            sensor_latency=sensor_latency,
            interpolator_fn=sensor.linear_obs_blender)

        #Desired Velocity Variables
        self.des_velocity = np.array([0, 0], dtype=np.float32)
        self._k_steps = k_steps
        self._step_count = 0
        self._dev_norm = dev_norm

        #Velocity Variables
        self._last_base_position = None

        self._dtype = np.float32
        self._observation_space = self._stack_space(
            gym.spaces.Box(
                low=lower_bound,
                high=upper_bound,
                dtype=self._dtype
            )
        )

    def _reset_des_vel(self):
        self.des_velocity = np.zeros(2, dtype=np.float32)
        self.normed_des = np.zeros(2, dtype=np.float32)
        self.des_speed = 0.0

    def _reset_vel_calc(self):
        self._last_base_position = np.array(self._robot.base_position)[:2]

    def on_reset(self, env):
        self._env = env
        self._observation_buffer.reset()
        
        self._reset_des_vel()
        self._reset_vel_calc()

        self.on_new_observation()

    def velocity_update(self):
        #Every k timesteps, perturb the desired velocity vector in 
        random_vec = np.random.uniform(low=-1.0, high=1.001, size=2)
        vec_norm = np.linalg.norm(random_vec)
        if vec_norm != 0:
            random_vec = random_vec / vec_norm
        pass

        random_vec = self._dev_norm * random_vec

        #print(f"-------\n{random_vec}")

        self.des_velocity = self.des_velocity + random_vec
    
    def get_robot_velocity(self):
        
        pass

    def _get_original_observation(self):
        des_vel = np.copy(self.des_velocity)
        current_vel = np.array(self._robot.base_position)[:2] - self._last_base_position

        ret = np.concatenate((des_vel, current_vel), axis=0)

        self._last_base_position = self._last_base_position
        if self._step_count % self._k_steps:
            self.velocity_update()
        
        self._step_count += 1
        return self._robot.timestamp, ret

    