"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import distrax
from flax.core.frozen_dict import FrozenDict

from jaxrl2.agents.agent import Agent
from jaxrl2.agents.sac import SACLearner
from jaxrl2.agents.iql import IQLLearner

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

@functools.partial(jax.jit, static_argnames=("critic_apply_fn", "critic_reduction"))
def _calculate_q_jit(
    critic_apply_fn: Callable[..., distrax.Distribution],
    critic_params: Params,
    observations: np.ndarray,
    actions:jnp.ndarray,
    critic_reduction: str
) -> jnp.ndarray:
    qs = critic_apply_fn(
        {"params": critic_params}, observations, actions
    )
    if critic_reduction == "min":
        q = qs.min(axis=0)
    elif critic_reduction == "mean":
        q = qs.mean(axis=0)
    else:
        raise NotImplemented()
    return q

class SACBasedPEXLearner(Agent):
    def __init__(
        self,
        seed: int,
        sac_agent: SACLearner,
        iql_agent: IQLLearner,
        inv_temperature: float,
        transfer_critic: bool = False,
        copy_to_target: bool = False,
    ):
        """
        An implementation of the version of PEX using SAC as the online backbone described in https://arxiv.org/pdf/2302.00935.pdf
        The current implementation assumes that the offline training is done using IQL.
        """
        
        rng = jax.random.PRNGKey(seed)
        rng, sac_eps_sample_key, pex_eps_sample_key = jax.random.split(rng, 3)

        self.sac_agent = sac_agent
        self.iql_agent = iql_agent
        self.inv_temperature = inv_temperature
        self.transfer_critic = transfer_critic
        self.copy_to_target = copy_to_target
        
        if self.transfer_critic:
            # transfer the offline-learned critic to the SAC agent for online finetunning
            self.sac_agent._critic.replace(params=self.iql_agent._critic.params)
            if self.copy_to_target: # copy the offline-learned critic to the target critic
                self.sac_agent._target_critic_params = copy.deepcopy(self.iql_agent._critic.params)
            else: # transfer the offline-learned target critic to the SAC agent for online finetunning
                self.sac_agent._target_critic_params = copy.deepcopy(self.iql_agent._target_critic_params)

        self._rng = rng
        self._sac_eps_sample_key = sac_eps_sample_key
        self._pex_eps_sample_key = pex_eps_sample_key

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        sample_epsilon = 0.1
        observations = np.expand_dims(observations, axis=0)
        a1 = jnp.array(self.iql_agent.eval_actions(observations))

        a2 = jnp.array(self.sac_agent.sample_actions(observations))
        greedy_a2 = jnp.array(self.sac_agent.eval_actions(observations))
        self._sac_eps_sample_key, sample_key = jax.random.split(self._sac_eps_sample_key)
        sac_greedy_mask = jax.random.uniform(sample_key, shape=(a2.shape[0],)) > sample_epsilon
        a2 = a2.at[sac_greedy_mask].set(greedy_a2[sac_greedy_mask])
        
        q1 = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            a1,
            self.iql_agent.critic_reduction
        )
        q2 = _calculate_q_jit(
            self.sac_agent._critic.apply_fn,
            self.sac_agent._critic.params,
            observations,
            a2,
            self.sac_agent.critic_reduction
        )

        q = jnp.stack([q1, q2], axis=-1)
        logits = q * self.inv_temperature
        w_dist = distrax.Categorical(logits=logits)
        
        greedy_policy_selection = w_dist.mode()
        self._pex_eps_sample_key, sample_key = jax.random.split(self._pex_eps_sample_key)
        ps_greedy_mask = jax.random.uniform(sample_key, shape=(greedy_policy_selection.shape[0],)) > sample_epsilon
        
        self._rng, key = jax.random.split(self._rng)
        w = w_dist.sample(seed=key)
        w = w.at[ps_greedy_mask].set(greedy_policy_selection[ps_greedy_mask])

        w = jnp.expand_dims(w, axis=-1)
        actions = (1 - w) * a1 + w * a2

        return np.asarray(jnp.squeeze(actions, 0))

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.expand_dims(observations, axis=0)
        a1 = jnp.array(self.iql_agent.eval_actions(observations))
        a2 = jnp.array(self.sac_agent.sample_actions(observations))

        q1 = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            a1,
            self.iql_agent.critic_reduction
        )
        q2 = _calculate_q_jit(
            self.sac_agent._critic.apply_fn,
            self.sac_agent._critic.params,
            observations,
            a2,
            self.sac_agent.critic_reduction
        )

        q = jnp.stack([q1, q2], axis=-1)
        logits = q * self.inv_temperature
        w_dist = distrax.Categorical(logits=logits)
        self._rng, key = jax.random.split(self._rng)
        w = w_dist.sample(seed=key)

        w = jnp.expand_dims(w, axis=-1)
        actions = (1 - w) * a1 + w * a2

        return np.asarray(jnp.squeeze(actions, 0))

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        info = self.sac_agent.update(batch)
        return info

    # This function calculate the log_probs from sac_agent instead of the PEX policy set
    # since the distribution of the PEX policy set can not be derived.
    # The reason to have this member function is to make the implementation of PEX agent class
    # complete; otherwise, the eval_log_probs() of the base class, Agent, will trigger an error
    # of unavailable class dataset member, _actor.
    def eval_log_probs(self, batch: DatasetDict) -> float:
        return self.sac_agent.eval_log_probs(batch)