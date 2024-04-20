# for saving JAX models
import copy
import jax
from gym.utils import seeding
from flax.training import orbax_utils
import orbax.checkpoint

from jaxrl2.agents import SACLearner, DrQLearner, BCLearner, IQLLearner, SACBasedPEXLearner

# kwargs: random generators of environments and replay buffer sampling.
#           This is for resuming online training.
def save_agent(orbax_checkpointer, agent, i, ckpt_filepath, force=True):
    if isinstance(agent, SACLearner):
        save_SAC_agent(orbax_checkpointer, agent, i, ckpt_filepath, force)
    elif isinstance(agent, IQLLearner):
        save_IQL_agent(orbax_checkpointer, agent, i, ckpt_filepath, force)
    elif isinstance(agent, SACBasedPEXLearner):
        save_SACBasedPEX_agent(orbax_checkpointer, agent, i, ckpt_filepath, force)
    else:
        raise ValueError("The agent to save must be an instance of SACLearner or IQLearner")

def load_agent(orbax_checkpointer, agent, ckpt_filepath):
    if isinstance(agent, SACLearner):
        return load_SAC_agent(orbax_checkpointer, agent, ckpt_filepath)
    elif isinstance(agent, IQLLearner):
        return load_IQL_agent(orbax_checkpointer, agent, ckpt_filepath)
    else:
        raise ValueError("The agent to load must be an instance of SACLearner or IQLearner")
    

def save_SACBasedPEX_agent(orbax_checkpointer, agent, i, ckpt_filepath, force=True):
    ckpt = {
        "step": i,
        "sac_agent": agent.sac_agent,
        "iql_agent": agent.iql_agent,
        "inv_temperature": agent.inv_temperature,
        "sample_epsilon" : agent.sample_epsilon,
        "transfer_critic": agent.transfer_critic,
        "copy_to_target" : agent.copy_to_target,
        "rng": agent._rng,
        "sac_eps_sample_rng": agent._sac_eps_sample_rng,
        "pex_eps_sample_rng": agent._pex_eps_sample_rng,
    }

    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_filepath, ckpt, save_args=save_args, force=force)

def load_IQL_agent(orbax_checkpointer, agent, ckpt_filepath):
    target = {
        "step": 0,
        "sac_agent": agent.sac_agent,
        "iql_agent": agent.iql_agent,
        "inv_temperature": agent.inv_temperature,
        "sample_epsilon" : agent.sample_epsilon,
        "transfer_critic": agent.transfer_critic,
        "copy_to_target" : agent.copy_to_target,
        "rng": agent._rng,
        "sac_eps_sample_rng": agent._sac_eps_sample_rng,
        "pex_eps_sample_rng": agent._pex_eps_sample_rng,
    }

    ckpt_restored = orbax_checkpointer.restore(ckpt_filepath, item=target)
    agent.sac_agent = ckpt_restored["sac_agent"]
    agent.iql_agent = ckpt_restored["iql_agent"]
    agent.inv_temperature = ckpt_restored["inv_temperature"]
    agent.sample_epsilon  = ckpt_restored["sample_epsilon"]
    agent.transfer_critic = ckpt_restored["transfer_critic"]
    agent.copy_to_target = ckpt_restored["copy_to_target"]
    agent._rng = ckpt_restored["rng"]
    agent._sac_eps_sample_rng = ckpt_restored["sac_eps_sample_rng"]
    agent._pex_eps_sample_rng = ckpt_restored["pex_eps_sample_rng"]

def save_IQL_agent(orbax_checkpointer, agent, i, ckpt_filepath, force=True):
    ckpt = {
        "step": i,
        "actor": agent._actor,
        "critic": agent._critic,
        "value": agent._value,
        "target_critic_params": agent._target_critic_params,
        "rng": agent._rng,
    }

    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_filepath, ckpt, save_args=save_args, force=force)

def load_IQL_agent(orbax_checkpointer, agent, ckpt_filepath):
    target = {
        "step": 0,
        "actor": agent._actor,
        "critic": agent._critic,
        "value": agent._value,
        "target_critic_params": agent._target_critic_params,
        "rng": agent._rng,
    }

    ckpt_restored = orbax_checkpointer.restore(ckpt_filepath, item=target)
    agent._actor = ckpt_restored["actor"]
    agent._critic = ckpt_restored["critic"]
    agent._value = ckpt_restored["value"]
    agent._target_critic_params = ckpt_restored["target_critic_params"]
    agent._rng = ckpt_restored["rng"]

def save_SAC_agent(orbax_checkpointer, agent, i, ckpt_filepath, force=True):
    ckpt = {
        "step": i,
        "actor": agent._actor,
        "critic": agent._critic,
        "temp": agent._temp,
        "target_critic_params": agent._target_critic_params,
        "rng": agent._rng,
    }

    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_filepath, ckpt, save_args=save_args, force=force)

def load_SAC_agent(orbax_checkpointer, agent, ckpt_filepath):
    target = {
        "step": 0,
        "actor": agent._actor,
        "critic": agent._critic,
        "temp": agent._temp,
        "target_critic_params": agent._target_critic_params,
        "rng": agent._rng,
    }

    ckpt_restored = orbax_checkpointer.restore(ckpt_filepath, item=target)
    agent._actor = ckpt_restored["actor"]
    agent._critic = ckpt_restored["critic"]
    agent._temp = ckpt_restored["temp"]
    agent._target_critic_params = ckpt_restored["target_critic_params"]
    agent._rng = ckpt_restored["rng"]

def initialize_SAC_agent_from_IQL_agent(orbax_checkpointer, sac_agent, iql_ckpt_filepath):
    #orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_restored = orbax_checkpointer.restore(iql_ckpt_filepath)
    sac_agent._actor.replace(params=ckpt_restored["actor"]["params"])
    sac_agent._critic.replace(params=ckpt_restored["critic"]["params"])
    sac_agent._target_critic_params = copy.deepcopy(ckpt_restored["critic"]["params"])
    sac_agent._rng = ckpt_restored["rng"] 

def equal_SAC_agents(agent1, agent2):
    equal_actors = jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent1._actor.params, agent2._actor.params))
    equal_critics = jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent1._critic.params, agent2._critic.params))
    equal_temps = jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent1._temp.params, agent2._temp.params))
    equal_target_critic_params = jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent1._target_critic_params, agent2._target_critic_params))
    equal_rngs = jax.tree_util.tree_all(jax.tree_map(lambda x, y: (x == y).all(), agent1._rng, agent2._rng))
    return (equal_actors and equal_critics and equal_temps and equal_target_critic_params and equal_rngs)
