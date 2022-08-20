from gym_jiminy.envs import ANYmalJiminyEnv, ANYmalPDControlJiminyEnv
import math

class ConfigurableANYmal(ANYmalJiminyEnv):
    def compute_reward(self, info):
        return self.reward_function(self,info)
    
    def set_reward_function(self, reward_function):
        self.reward_function = reward_function

    def is_done(self):
        return self.system_state.q[2] < self._height_neutral * 0.4


# returns -10 if the robot has fallen over else 1
def simple_standing_reward(self, info):
    return -10 if self.is_done() else 1

# returns -10 if the robot has fallen over else the robot's speed in the X direction (starts facing towards X)
def forward_reward(self, info):
    return -10 if self.is_done() else self.state[1][0]

# returns a reward function that gives positive reward for moving forward smoothly
def smooth_forward_reward(jerkiness_penalty):
    def _reward(self, info):
        jerkiness = jerkiness_penalty * math.sqrt(sum([self.state[1][x] ** 2 for x in range(1,6)]))
        return -10 if self.is_done() else self.state[1][0] - jerkiness
    return _reward

def smooth_forward_reward_2(jerkiness_penalty):
    def _reward(self, info):
        jerkiness = jerkiness_penalty * math.sqrt(sum([self.state[1][x] ** 2 for x in range(1,6)]))
        return -10 if self.is_done() else (10 + self.state[1][0]) / (1 + jerkiness)
    return _reward

def smooth_forward_reward_3(jerkiness_penalty):
    def _reward(self, info):
        height = self.state[0][2] / self._height_neutral if self.state[0][2] < self._height_neutral else 1  
        jerkiness = jerkiness_penalty * math.sqrt(sum([self.state[1][x] ** 2 for x in range(1,6)]))
        return -10 if self.is_done() else (10 + self.state[1][0])* height / (1 + jerkiness)
    return _reward