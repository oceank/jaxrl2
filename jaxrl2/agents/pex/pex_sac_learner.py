"""Implementations of algorithms for continuous control."""

import copy
import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import distrax
from flax.core.frozen_dict import FrozenDict

from jaxrl2.agents.common import eval_actions_jit, sample_actions_jit
from jaxrl2.agents.agent import Agent
from jaxrl2.agents.sac import SACLearner
from jaxrl2.agents.iql import IQLLearner

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

@functools.partial(jax.jit, static_argnames="inv_temperature")
def _select_policy_to_act_jit(
    rng: PRNGKey,
    q1: jnp.ndarray,
    q2: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    inv_temperature: float
) -> Tuple[PRNGKey, jnp.ndarray]:
    q = jnp.stack([q1, q2], axis=-1)
    logits = q * inv_temperature
    w_dist = distrax.Categorical(logits=logits)
    rng, key = jax.random.split(rng)
    w = w_dist.sample(seed=key)

    w = jnp.expand_dims(w, axis=-1)
    actions = (1 - w) * a1 + w * a2

    return rng, actions 

@functools.partial(jax.jit, static_argnames="sample_epsilon")
def _greedy_eps_sample_action(
    rng: PRNGKey,
    sampled_actions: jnp.ndarray,
    greedy_actions: jnp.ndarray,
    sample_epsilon: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)
    #greedy_mask = jax.random.uniform(key, shape=(greedy_actions.shape[0],)) > sample_epsilon
    #sampled_actions = sampled_actions.at[greedy_mask].set(greedy_actions[greedy_mask])
    #print(f"greedy_actions: {greedy_actions.shape}")
    greedy_mask = jax.random.uniform(key, shape=(greedy_actions.shape[0],)) > sample_epsilon
    #greedy_mask = jnp.expand_dims(greedy_mask, axis=0)
    #greedy_mask = jnp.repeat(greedy_mask, greedy_actions.shape[1], axis=1)
    sampled_actions = jnp.where(
        greedy_mask[:, None],
        greedy_actions,
        sampled_actions,
    )
 
    return rng, sampled_actions

@functools.partial(jax.jit, static_argnames=("inv_temperature", "sample_epsilon"))
def _select_policy_with_greedy_eps_to_act_jit(
    rng_policy_sampling: PRNGKey,
    rng_greedy_eps: PRNGKey,
    q1: jnp.ndarray,
    q2: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    inv_temperature: float,
    sample_epsilon: float,
) -> Tuple[PRNGKey, PRNGKey, jnp.ndarray]:
    q = jnp.stack([q1, q2], axis=-1)
    logits = q * inv_temperature
    w_dist = distrax.Categorical(logits=logits)

    # sampled policy
    rng_policy_sampling, key = jax.random.split(rng_policy_sampling)
    sampled_policy = w_dist.sample(seed=key)
    # greedy policy
    greedy_policy = w_dist.mode()
    # determine the final selected policy
    rng_greedy_eps, policy_selection = _greedy_eps_sample_action(
        rng_greedy_eps, sampled_policy[:, None], greedy_policy[:, None], sample_epsilon)
    policy_selection = jnp.squeeze(policy_selection, 0)

    policy_selection = jnp.expand_dims(policy_selection, axis=-1)
    actions = (1 - policy_selection) * a1 + policy_selection * a2

    return rng_policy_sampling, rng_greedy_eps, actions 


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
        sample_epsilon = 0.1,
    ):
        """
        An implementation of the version of PEX using SAC as the online backbone described in https://arxiv.org/pdf/2302.00935.pdf
        The current implementation assumes that the offline training is done using IQL.
        """
        
        rng = jax.random.PRNGKey(seed)
        rng, sac_eps_sample_rng, pex_eps_sample_rng = jax.random.split(rng, 3)

        self.sac_agent = sac_agent
        self.iql_agent = iql_agent
        self.inv_temperature = inv_temperature
        self.transfer_critic = transfer_critic
        self.copy_to_target = copy_to_target
        self.sample_epsilon = sample_epsilon

        if self.transfer_critic:
            # transfer the offline-learned critic to the SAC agent for online finetunning
            self.sac_agent._critic.replace(params=self.iql_agent._critic.params)
            if self.copy_to_target: # copy the offline-learned critic to the target critic
                self.sac_agent._target_critic_params = copy.deepcopy(self.iql_agent._critic.params)
            else: # transfer the offline-learned target critic to the SAC agent for online finetunning
                self.sac_agent._target_critic_params = copy.deepcopy(self.iql_agent._target_critic_params)

        self._rng = rng
        self._sac_eps_sample_rng = sac_eps_sample_rng
        self._pex_eps_sample_rng = pex_eps_sample_rng

    def get_actions_of_policy_set(self, observations: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        iql_actions = eval_actions_jit(
            self.iql_agent._actor.apply_fn,
            self.iql_agent._actor.params,
            observations
        )
        sac_rng, sac_actions = sample_actions_jit(
            self.sac_agent._rng,
            self.sac_agent._actor.apply_fn,
            self.sac_agent._actor.params,
            observations
        )
        self.sac_agent_rng = sac_rng
        return iql_actions, sac_actions

    def get_greedy_action_of_sac_agent(self, observations: np.ndarray) -> jnp.ndarray:
        actions = eval_actions_jit(
            self.sac_agent._actor.apply_fn,
            self.sac_agent._actor.params,
            observations
        )
        return actions

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.expand_dims(observations, axis=0)
 
        iql_actions, sac_sampled_actions = self.get_actions_of_policy_set(observations)
        sac_greedy_actions = self.get_greedy_action_of_sac_agent(observations)

        sac_eps_sample_rng, sac_actions = _greedy_eps_sample_action(
            self._sac_eps_sample_rng,
            sac_sampled_actions,
            sac_greedy_actions,
            self.sample_epsilon
        )
        self._sac_eps_sample_rng = sac_eps_sample_rng
       
        iql_q = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            iql_actions,
            self.iql_agent.critic_reduction
        )
        sac_q = _calculate_q_jit(
            self.sac_agent._critic.apply_fn,
            self.sac_agent._critic.params,
            observations,
            sac_actions,
            self.sac_agent.critic_reduction
        )

        rng, pex_eps_sample_rng, actions = _select_policy_with_greedy_eps_to_act_jit(
            self._rng,
            self._pex_eps_sample_rng,
            iql_q,
            sac_q,
            iql_actions,
            sac_actions,
            self.inv_temperature,
            self.sample_epsilon
        )

        self._rng = rng
        self._pex_eps_sample_rng = pex_eps_sample_rng
        actions = jnp.squeeze(actions, 0)

        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.expand_dims(observations, axis=0)

        iql_actions, sac_actions = self.get_actions_of_policy_set(observations)
 
        iql_q = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            iql_actions,
            self.iql_agent.critic_reduction
        )
        sac_q = _calculate_q_jit(
            self.sac_agent._critic.apply_fn,
            self.sac_agent._critic.params,
            observations,
            sac_actions,
            self.sac_agent.critic_reduction
        )

        rng, actions = _select_policy_to_act_jit(
            self._rng,
            iql_q, sac_q,
            iql_actions, sac_actions,
            self.inv_temperature
        )

        self._rng = rng
        actions = jnp.squeeze(actions, 0)

        return np.asarray(actions)

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
