from typing import List, Iterable, Optional
from functools import reduce

import gym
import numpy as np
from flax.core import frozen_dict
from jaxrl2.data import Dataset

import scipy.special
from sklearn.linear_model import LinearRegression

# Paper: HARNESSING MIXED OFFLINE REINFORCEMENT LEARNING DATASETS VIA TRAJECTORY WEIGHTING
# Reference Code (April 1 2024): https://github.com/Improbable-AI/harness-offline-rl/blob/master/d3rlpy_examples/weighted_datasets.py
class Wrapper:
    def  __init__(self, base_obj):
        self.base_obj = base_obj
    
    def __getattr__(self, name):
        return getattr(self.base_obj, name)

class ReturnWeightedReplayBufferWrapper(Wrapper):

    def __init__(self, 
            dataset: Dataset, 
            alpha: float = 0.1, # Temperature of the softmax. The higher, the closer to uniform distribution.
            cache_size: int = int(1e6),
        ):
        super().__init__(dataset)
        self.alpha = alpha
        self.cache_size = cache_size # batch_size*num_of_gradient_steps

        (self.episode_starts, self.episode_ends, self.episode_returns,) = dataset._trajectory_boundaries_and_returns(record_last_traj_return_if_not_done=True)
        #self.episode_starts = episode_starts
        #self.episode_ends = episode_ends
        #self.episode_returns = episode_returns
        self.transition_counts_per_episode = [(end - start) for start, end in zip(self.episode_starts, self.episode_ends)]
        self.total_transitions = sum(self.transition_counts_per_episode)
        
        self.sample_probs = self._compute_sample_probs()
        assert self.total_transitions == len(self.sample_probs)

        # Compute the probability of each episode for visualizing reweigted dataset
        self.episode_probs = np.array([self.sample_probs[start:end].sum() for start, end in zip(self.episode_starts, self.episode_ends)])
        #self.episode_probs /= self.episode_probs.sum() # normalize to avoid numerical errors
        
        self._generate_cache(cache_size)

    def get_episode_returns(self):
        return self.episode_returns

    def minmax_normalization(self, x):
        if x.max() == x.min():
            return x - x.min()
        else: # x.max() != x.min()
            return (x - x.min()) / (x.max() - x.min())

    def _compute_sample_probs(self):
        G = np.asarray(self.episode_returns)
        G = self.minmax_normalization(G) # section 4.2 in the paper
        T = np.asarray(self.transition_counts_per_episode)
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        w_it = (G_it - G_it.min()) / (G_it.max() - G_it.min())
        w_it = scipy.special.softmax(G_it / self.alpha)       
        return w_it

    def _generate_cache(self, cache_size: int):
        """
        Sample a large batch of transition indices with weighted sampling.
        This is needed because calling `np.random.choice` with `p` is slow.
        """
        # self.np_random is a property of self.base_obj, an instance of Dataset
        self._cache_indices = self.np_random.choice(
                range(self.sample_probs.shape[0]),
                cache_size,
                p=self.sample_probs)
        self._cache_pointer = 0

    def _pop_sample_index_from_cache(self, num_samples):
        indx = np.array([-1]*num_samples)
        for i in range(num_samples):
            indx[i] = self._cache_indices[self._cache_pointer]
            self._cache_pointer += 1
             # Refresh cache when no element in the cache
            if self._cache_pointer == self._cache_indices.shape[0]:
                self._generate_cache(self.cache_size)

        return indx

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> frozen_dict.FrozenDict:
        if indx is None:
            indx = self._pop_sample_index_from_cache(batch_size)
        return self.base_obj.sample(batch_size, keys, indx)
        


class AdvantageWeightedReplayBufferWrapper(ReturnWeightedReplayBufferWrapper):
       
    def _compute_sample_probs(self):
        G = np.asarray(self.episode_returns)
        G = self.minmax_normalization(G) # section 4.2 in the paper
        T = np.asarray(self.transition_counts_per_episode)
        G_it = np.asarray(reduce(lambda x, y: x + y, [[G_i] * T_i for G_i, T_i in zip(G, T)]))
        # dataset_dict is a property of self.base_obj, an instance of Dataset
        s0 = np.stack([self.dataset_dict['observations'][start_obs_idx] for start_obs_idx in self.episode_starts])
        V = LinearRegression().fit(s0, G).predict(s0)
        V_it = np.asarray(reduce(lambda x, y: x + y, [[V_i] * T_i for V_i, T_i in zip(V, T)]))
        A_it = G_it - V_it
        A_it = (A_it - A_it.min()) / (A_it.max() - A_it.min())
        w_it = scipy.special.softmax(A_it / self.alpha)
        w_it /= w_it.sum() # Avoid numerical errors
        return w_it