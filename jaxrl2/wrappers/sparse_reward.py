import gym

class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_accumulate_steps=1):
        super().__init__(env)
        self.reward_accumulate_steps = reward_accumulate_steps
        self.current_accumulated_reward = 0.0
        self.current_step_in_episode = 0

    def step(self, action):
        observation, reward, done, info = super().step(action) #self.env.step(action)

        # Accumulate the reward
        self.current_accumulated_reward += reward
        self.current_step_in_episode += 1

        if (self.current_step_in_episode % self.reward_accumulate_steps == 0) or done:
            accumulated_reward = self.current_accumulated_reward
            self.current_accumulated_reward = 0.0

            if done:
                self.current_step_in_episode = 0

        else: # Otherwise, continue accumulating rewards
            accumulated_reward = 0.0

        return observation, accumulated_reward, done, info

    def reset(self, **kwargs):
        self.current_accumulated_reward = 0.0
        self.current_step_in_episode = 0
        return super().reset(**kwargs) #self.env.reset(**kwargs)