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
from jaxrl2.agents.pex.actor_updater import update_actor
from jaxrl2.agents.pex.critic_updater import update_q, update_v
from jaxrl2.utils.target_update import soft_target_update
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from flax.training.train_state import TrainState





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

@functools.partial(jax.jit, static_argnames="critic_reduction")
def _update_jit(
    pex_actions: jnp.ndarray,
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    value: TrainState,
    batch: TrainState,
    discount: float,
    tau: float,
    expectile: float,
    A_scaling: float,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:

    target_critic = critic.replace(params=target_critic_params)
    new_value, value_info = update_v(
        target_critic, value, batch, expectile, critic_reduction
    )
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        pex_actions, key, actor, target_critic, new_value, batch, A_scaling, critic_reduction
    )

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic_params = soft_target_update(
        new_critic.params, target_critic_params, tau
    )

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic_params,
        new_value,
        {**critic_info, **value_info, **actor_info},
    )


class IQLBasedPEXLearner(Agent):
    def __init__(
        self,
        seed: int,
        iql_online_agent: IQLLearner,
        iql_agent: IQLLearner,
        inv_temperature: float,
        transfer_critic: bool = False,
        copy_to_target: bool = False,
        sample_epsilon = 0.1,
    ):
        """
        An implementation of the version of IQL using SAC as the online backbone described in https://arxiv.org/pdf/2302.00935.pdf
        The current implementation assumes that the offline training is done using IQL.
        """
        
        rng = jax.random.PRNGKey(seed)
        rng, online_policy_eps_sample_rng, pex_eps_sample_rng = jax.random.split(rng, 3)

        self.iql_online_agent = iql_online_agent
        self.iql_agent = iql_agent
        self.inv_temperature = inv_temperature
        self.transfer_critic = transfer_critic
        self.copy_to_target = copy_to_target
        self.sample_epsilon = sample_epsilon

        if self.transfer_critic:
            # transfer the offline-learned critic to the SAC agent for online finetunning
            self.iql_online_agent._critic.replace(params=self.iql_agent._critic.params)
            self.iql_online_agent._value.replace(params=self.iql_agent._value.params)

            if self.copy_to_target: # copy the offline-learned critic to the target critic
                self.iql_online_agent._target_critic_params = copy.deepcopy(self.iql_agent._critic.params)
            else: # transfer the offline-learned target critic to the SAC agent for online finetunning
                self.iql_online_agent._target_critic_params = copy.deepcopy(self.iql_agent._target_critic_params)

        self._rng = rng
        self._online_policy_eps_sample_rng = online_policy_eps_sample_rng
        self._pex_eps_sample_rng = pex_eps_sample_rng

    def get_actions_of_policy_set(self, observations: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        iql_actions = eval_actions_jit(
            self.iql_agent._actor.apply_fn,
            self.iql_agent._actor.params,
            observations
        )
        iql_online_agent_rng, iql_online_actions = sample_actions_jit(
            self.iql_online_agent._rng,
            self.iql_online_agent._actor.apply_fn,
            self.iql_online_agent._actor.params,
            observations
        )
        self.iql_online_agent._rng = iql_online_agent_rng
        return iql_actions, iql_online_actions

    def get_greedy_action_of_online_agent(self, observations: np.ndarray) -> jnp.ndarray:
        actions = eval_actions_jit(
            self.iql_online_agent._actor.apply_fn,
            self.iql_online_agent._actor.params,
            observations
        )
        return actions

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.expand_dims(observations, axis=0)
 
        iql_actions, iql_online_sampled_actions = self.get_actions_of_policy_set(observations)
        iql_online_greedy_actions = self.get_greedy_action_of_online_agent(observations)

        online_policy_eps_sample_rng, iql_online_actions = _greedy_eps_sample_action(
            self._online_policy_eps_sample_rng,
            iql_online_sampled_actions,
            iql_online_greedy_actions,
            self.sample_epsilon
        )
        self._online_policy_eps_sample_rng = online_policy_eps_sample_rng
       
        iql_q = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            iql_actions,
            self.iql_agent.critic_reduction
        )
        iql_online_q = _calculate_q_jit(
            self.iql_online_agent._critic.apply_fn,
            self.iql_online_agent._critic.params,
            observations,
            iql_online_actions,
            self.iql_online_agent.critic_reduction
        )

        rng, pex_eps_sample_rng, actions = _select_policy_with_greedy_eps_to_act_jit(
            self._rng,
            self._pex_eps_sample_rng,
            iql_q,
            iql_online_q,
            iql_actions,
            iql_online_actions,
            self.inv_temperature,
            self.sample_epsilon
        )

        self._rng = rng
        self._pex_eps_sample_rng = pex_eps_sample_rng
        actions = jnp.squeeze(actions, 0)

        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.expand_dims(observations, axis=0)

        iql_actions, iql_online_actions = self.get_actions_of_policy_set(observations)
 
        iql_q = _calculate_q_jit(
            self.iql_agent._critic.apply_fn,
            self.iql_agent._critic.params,
            observations,
            iql_actions,
            self.iql_agent.critic_reduction
        )
        iql_online_q = _calculate_q_jit(
            self.iql_online_agent._critic.apply_fn,
            self.iql_online_agent._critic.params,
            observations,
            iql_online_actions,
            self.iql_online_agent.critic_reduction
        )

        rng, actions = _select_policy_to_act_jit(
            self._rng,
            iql_q, iql_online_q,
            iql_actions, iql_online_actions,
            self.inv_temperature
        )

        self._rng = rng
        actions = jnp.squeeze(actions, 0)

        return np.asarray(actions)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        pex_actions = self.sample_actions(batch["observations"])
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_value,
            info,
        ) = _update_jit(
            pex_actions,
            self.iql_online_agent._rng,
            self.iql_online_agent._actor,
            self.iql_online_agent._critic,
            self.iql_online_agent._target_critic_params,
            self.iql_online_agent._value,
            batch,
            self.iql_online_agent.discount,
            self.iql_online_agent.tau,
            self.iql_online_agent.expectile,
            self.iql_online_agent.A_scaling,
            self.iql_online_agent.critic_reduction,
        )

        self.iql_online_agent._rng = new_rng
        self.iql_online_agent._actor = new_actor
        self.iql_online_agent._critic = new_critic
        self.iql_online_agent._target_critic_params = new_target_critic
        self.iql_online_agent._value = new_value

        return info

    def eval_log_probs(self, batch: DatasetDict) -> float:
        eval_actions = self.eval_actions(batch["observations"])
        dist = self.iql_online_agent._actor.apply_fn(
            {"params": self.iql_online_agent._actor.params},
            batch["observations"],
            training=False,
        )
        log_probs = dist.log_prob(eval_actions)
        
        return log_probs.mean()
