import gym
from gym.wrappers.flatten_observation import FlattenObservation

from jaxrl2.wrappers.single_precision import SinglePrecision
from jaxrl2.wrappers.universal_seed import UniversalSeed

from jaxrl2.wrappers.sparse_reward import SparseRewardWrapper


def wrap_gym(env: gym.Env, rescale_actions: bool = True, reward_accumulate_steps=1) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    env = SparseRewardWrapper(env, reward_accumulate_steps=reward_accumulate_steps)

    return env
