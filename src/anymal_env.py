from gym_jiminy.envs import ANYmalJiminyEnv
import math

class ConfigurableANYmal(ANYmalJiminyEnv):
    def compute_reward(self, info):
        return self.reward_function(self,info)
    
    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

# returns -10 if the robot has fallen over else 1
def simple_standing_reward(self, info):
    return -10 if self.is_done() else 1

# returns -10 if the robot has fallen over else the robot's speed in the X direction (starts facing towards X)
def forward_reward(self, info):
    return -10 if self.is_done() else self.state[1][0]

# returns a reward function that gives positive reward for moving forward smoothly
def smooth_forward_reward(jerkiness_penalty):
    def _reward(self, info):
        jerkiness = jerkiness_penalty * math.sqrt(self.state[1][1]**2 + self.state[1][2]**2)
        return -10 if self.is_done() else self.state[1][0] - jerkiness
    return _reward
