from gym import Env, spaces
import numpy as np

class SalpakanEnv(Env):

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(9, 8, 5), dtype=np.float16)
        self.action_space = spaces.Discrete(85)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)