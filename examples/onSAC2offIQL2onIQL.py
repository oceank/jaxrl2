#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
import math
import json
import csv 
import gym
import numpy as np
import tqdm
import wandb
import jax
import orbax.checkpoint
import optax
from optax._src import base
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict
import matplotlib.pyplot as plt
from datetime import datetime

from jaxrl2.agents import SACLearner, IQLLearner
from jaxrl2.data import ReplayBuffer, plot_episode_returns
from jaxrl2.utils.save_load_agent import load_SAC_agent, load_IQL_agent, save_agent, load_agent
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_expr_log import save_log

from tensorboardX import SummaryWriter

from functools import singledispatch


# Note1:
# the input parameter, agent, is updated by replacing its data member, "_actor", with a new actor
# whose optimizer and its state is updated as described in Note2.
# Note2:
# the actor of the iql agent was trained using an adam optimizer with a cosin scheduler
# so, agent._actor.opt_state has two elements where the 2nd element for ScaledByScheduler.
# When swithcting to a new adam optimizer with a fixed learning rate, the 2nd element of
# the optimizer state, opt_state, should be replaced by base.EmptyState()
def update_iql_actor_optimizer_with_fixed_lr_for_finetunning(agent, actor_lr):
    new_opt_state=(agent._actor.opt_state[0], base.EmptyState())
    new_optim=optax.adam(learning_rate=actor_lr)
    agent._actor = agent._actor.replace(opt_state=new_opt_state, tx=new_optim)


# Note1:
# the input parameter, agent, is updated by replacing its data member, "_actor", with a new actor
# whose optimizer and its state is updated as described in Note2.
# Note2:
# Create the new cosine scheduler that reaches a learning rate of 'actor_lr' after 'agent._actor.opt_state[0].count' iterations.
# And the total decay steps of the new cosine scheduler is 'agent._actor.opt_state[0].count' + 'finetunning_budget'
# the new cosine scheduler will reach a learning rate no smaller than 'min_finetunning_lr'
# Cosine Scheduler: alpha+(1+cos(pi*current_iteration/total_decay_steps)^p)*initial_lr*(1-alpha)/2
#   alpha: the minimum value of learning rate
#   initial_lr: the initial value of learning rate
#   p: it is set to 1 here
def update_iql_actor_optimizer_with_new_cosine_scheduler_for_finetunning(agent, actor_lr, min_finetunning_lr, finetunning_budget):
    count = agent._actor.opt_state[0].count
    total_decay_steps = count+finetunning_budget
    alpha = min_finetunning_lr
    # create the new cosine scheduler
    cosine_weight = (1.0+math.cos((count/total_decay_steps)*math.pi))*(1.0-alpha)/2
    initial_lr = (actor_lr-alpha)/cosine_weight
    new_cosine_scheduler=optax.cosine_decay_schedule(initial_value=initial_lr, decay_steps=total_decay_steps, alpha=alpha)
    # update the scheduler of the actor in the iql agent 
    agent._actor = agent._actor.replace(tx=optax.adam(learning_rate=new_cosine_scheduler))


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

max_steps_per_episode_dict = {
    "kitchen-complete-v0": 280,
    "kitchen-mixed-v0": 280,
    "kitchen-new-v0": 280,
    "pen-expert-v0": 100,
    "door-expert-v0": 200,
    "hammer-expert-v0": 200,
    "Ant-v2": 1000,
    "Hopper-v2": 1000,
    "Walker2d-v2": 1000,
    "maze2d-umaze-v1": 300,
    "maze2d-medium-v1": 600,
    "maze2d-large-v1": 800,
}

def save_machine_info(filename):
    """
    Saves the output of hostname and nvidia-smi to a file.
    Checks if JAX is using the GPU.

    Args:
        filename: The name of the file to save the output to.
    """
    commands = ["hostname", "echo", "nvidia-smi"]
    with open(filename, "w") as f:
        # Save the hostname, NIVDIA driver version and CUDA version 
        for command in commands:
            output = subprocess.check_output(command)
            f.write(output.decode("utf-8"))
        # Check if JAX is using the GPU
        f.write("jax.default_backend():\n")
        result = jax.default_backend()
        f.write(f"==>{result}\n")
        f.write("jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0]):\n")
        result = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        f.write(f"==>{result}\n")

def save_eval_info(
    filename,
    online_budget_total,
    online_eval_budget_ratio, offline_eval_budget_ratio,
    eval_start_step_ratio, online_stop_step, max_offline_gradient_steps,
    eval_episodes_per_evaluation,
    online_evals, online_eval_steps,
    offline_evals, offline_eval_steps,):
    """
    Saves the evaluation information to a file.
    Args:
        filename: The name of the file to save the output to.
    """

    with open(filename, "w") as f:
        f.write("Evaluation Information\n")
        f.write(f"online_budget_total: {online_budget_total}\n")
        f.write(f"online_eval_budget_ratio: {online_eval_budget_ratio}\n")
        f.write(f"offline_eval_budget_ratio: {offline_eval_budget_ratio}\n")
        f.write(f"eval_start_step_ratio: {eval_start_step_ratio}\n")
        f.write(f"online_stop_step: {online_stop_step}\n")
        f.write(f"max_offline_gradient_steps: {max_offline_gradient_steps}\n")
        f.write(f"eval_episodes_per_evaluation: {eval_episodes_per_evaluation}\n")
        f.write(f"online_evals: {online_evals}\n")
        f.write(f"online_eval_steps: {online_eval_steps}\n")
        f.write(f"offline_evals: {offline_evals}\n")
        f.write(f"offline_eval_steps: {offline_eval_steps}\n")

Training_Testing_Seed_Gap = 10000
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("max_offline_gradient_steps", int(5e5), "The number of gradient steps performed by the offline RL")

flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation. It is used to evaluate the final ouput policy")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Number of online interaction steps between two evaluations during online finetunning. Here, assume that no best pooling is applied for online finetunning process")
#flags.DEFINE_integer("eval_start_step", int(1e5), "The evaluations start at the step (either interaction step in online RL or gradient step in offline RL).")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer(
    "start_online_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("online_stop_step", int(8e5), "The step when the online learning stops")
flags.DEFINE_integer("online_budget_total", int(1e6), "The total budget of steps for online interactions")
flags.DEFINE_integer("eval_budget_base", int(1e6), "The number of steps for calculating the actual budget of steps for best pooling by multiplying it with a ratio.")


flags.DEFINE_integer("online_budget_per_offline_trigger", int(2e5), "The budget of online interactions for each offline procedure")
flags.DEFINE_integer("eval_episodes_per_evaluation", 3, "The number of episodes used for evaluation during each evaluation")
flags.DEFINE_integer("reward_accumulate_steps", 1, "Number of steps to accumulate the reward for sparsing the orignal reward.")

flags.DEFINE_float("online_eval_budget_ratio", 0.02, "The ratio of the budget used for evaluations during each online phase")
flags.DEFINE_float("offline_eval_budget_ratio", 0.01, "The ratio of the budget used for evaluations during each offline phase")
flags.DEFINE_float("eps_exploration", 0.0, "The probability of applying a random action instead of the action (the best action) by the reference agent")
flags.DEFINE_float("eval_start_step_ratio", 0.0, "The ratio of the learning step when evaluation distribution starts (either interaction step in online RL or gradient step in offline RL).")
flags.DEFINE_boolean("use_cosine_decay_for_finetunning", False, "use cosine decay scheduler for the learning rate used in the online finetunning after the offline IQL")
flags.DEFINE_boolean("use_fixed_lr_for_both_offline_and_finetunning", True, "both offline IQL and online finetunning IQL use the same fixed learning rate for the optimizer used in the actor")
#flags.DEFINE_boolean("use_all_data_for_offline_procedure", False, "use all collected trajectory data so far for the following offline RL; if False, only use the newly collected trajectory")
flags.DEFINE_boolean("apply_best_pooling", True, "use best pooling to select evalauted policies during training and return the best policy as the output policy of each procedure; otherwise, select the last policy during training as the output policy of each policy")
flags.DEFINE_boolean("normalize_eval_return", False, "normalize the evaluation result")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_offline_dataset", False, "Save the dataset used by the offline procedure")
config_flags.DEFINE_config_file(
    "config_online",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "config_offline",
    "configs/offline_config.py:iql_mujoco",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def main(_):
    # create the project directory
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    if "ras" in FLAGS.env_name:
        pieces = FLAGS.env_name.split("-")
        ras = int(pieces[-1][3:])
        FLAGS.reward_accumulate_steps = ras
        FLAGS.env_name = "-".join(pieces[:-1])

    project_name = f"{FLAGS.env_name}"
    if FLAGS.reward_accumulate_steps!=1:
        project_name += f"-ras{FLAGS.reward_accumulate_steps}"
    project_name += f"_seed{FLAGS.seed}_on2off2on_oss{(FLAGS.online_stop_step//1000)}k"
    if FLAGS.apply_best_pooling:
        project_name += f"_OnlineEvalBudget{FLAGS.online_eval_budget_ratio}_OfflineEvalBudget{FLAGS.offline_eval_budget_ratio}"
    else:
        project_name += "_NoBestPooling"
    if FLAGS.online_budget_per_offline_trigger==0: # no new data, just use the current buffer
        project_name += "_OfflineUseBuffer"
    else:
        project_name += f"_OfflineUseNewData{FLAGS.online_budget_per_offline_trigger//1000}kExplorationEPS{FLAGS.eps_exploration}"
    if FLAGS.use_cosine_decay_for_finetunning:
        project_name += "_FinetunningUseCosineScheduler"
    else:
        project_name += "_FinetunningUseFixedLR"
    project_name += f"_{expr_time_str}"
    project_dir = os.path.join(FLAGS.save_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # RL validation setups
    eval_episodes_per_evaluation = FLAGS.eval_episodes_per_evaluation
    ## online RL validation setup:
    ### Calculate the list of steps where the policies are saved and evaluated for optimal pooling. The steps in the list are evenly distributed, following a periodic pattern.
    online_eval_episodes = int(FLAGS.eval_budget_base * FLAGS.online_eval_budget_ratio / max_steps_per_episode_dict[FLAGS.env_name])
    online_evals = online_eval_episodes//eval_episodes_per_evaluation
    online_eval_start_step = int(FLAGS.eval_start_step_ratio*FLAGS.online_budget_total)
    online_eval_steps = list(
        range(
            FLAGS.online_stop_step,
            online_eval_start_step,
            -1*((FLAGS.online_stop_step-online_eval_start_step)//(online_evals-1))
            )
        )
    if (len(online_eval_steps) + 1) == online_evals:
        #left_evals = online_evals-len(online_eval_steps)
        if online_eval_start_step == 0: # the agent at the step 0 definitely has a randomly initialized policy. So, do not evaluate here
            online_eval_steps.append(online_eval_steps[-1]//2) # the step at the middle of step 0 and the the currently earliest evaluation step
        else:
            online_eval_steps.append(online_eval_start_step)

    if FLAGS.online_eval_budget_ratio == 0:
        online_eval_steps = [FLAGS.online_stop_step]

    ## offline RL validation setup: similar to the setup for online RL validation
    offline_eval_episodes = int(FLAGS.eval_budget_base * FLAGS.offline_eval_budget_ratio / max_steps_per_episode_dict[FLAGS.env_name])
    offline_evals = offline_eval_episodes//eval_episodes_per_evaluation
    offline_eval_start_step = int(FLAGS.eval_start_step_ratio*FLAGS.max_offline_gradient_steps)
    offline_eval_steps = list(
        range(
            FLAGS.max_offline_gradient_steps,
            offline_eval_start_step,
            -1*((FLAGS.max_offline_gradient_steps-offline_eval_start_step)//(offline_evals-1))
            )
        )
    if (len(offline_eval_steps) + 1) == offline_evals:
        if offline_eval_start_step == 0: # the agent at the step 0 definitely has a randomly initialized policy. So, do not evaluate here
            offline_eval_steps.append(offline_eval_steps[-1]//2) # the step at the middle of step 0 and the the currently earliest evaluation step
        else:
            offline_eval_steps.append(offline_eval_start_step)

    if FLAGS.offline_eval_budget_ratio == 0:
        offline_eval_steps = [FLAGS.max_offline_gradient_steps]

    # save the evaluation information
    save_eval_info(
        f"{project_dir}/eval_info.txt",
        FLAGS.online_budget_total,
        FLAGS.online_eval_budget_ratio, FLAGS.offline_eval_budget_ratio,
        FLAGS.eval_start_step_ratio, FLAGS.online_stop_step, FLAGS.max_offline_gradient_steps,
        eval_episodes_per_evaluation,
        online_evals, online_eval_steps,
        offline_evals, offline_eval_steps,
    )

    # save the machine's nvidia-smi output
    save_machine_info(f"{project_dir}/machine_info.txt")

    # save experiment configuration to file
    flags_dict = flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{project_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict[_[0]].items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4, default=to_serializable)

    # Initialize the logger for the experiment
    if FLAGS.wandb:
        summary_writer = wandb.init(project=project_name)
        summary_writer.config.update(FLAGS)
    else:
        summary_writer = SummaryWriter(project_dir, write_to_disk=True)

    # create the environments
    ## env for collecting new trajectories for offline procedures
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True, reward_accumulate_steps=FLAGS.reward_accumulate_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    ## eval_env is for policy evaluation during each offline procedure
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True, reward_accumulate_steps=FLAGS.reward_accumulate_steps)
    eval_env.seed(FLAGS.seed + Training_Testing_Seed_Gap)


    # Phase 1: Purely Online RL phase
    ## create the replay buffer for the purely online RL phase
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.online_budget_total
    )
    replay_buffer.seed(FLAGS.seed)

    ## create the SAC-based agent and initialize the orbax checkpointer for saving the agent periodically
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    kwargs_online = dict(FLAGS.config_online)
    agent_online = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs_online)

    online_eval_filepath = f"{project_dir}/eval_ave_episode_return_online{FLAGS.online_stop_step//1000}k.txt"
    with open(online_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tStep\tReturn\n")

    online_eval_file = open(online_eval_filepath, "a")

    start_step = 1
    max_steps = FLAGS.online_stop_step
    best_online_agent_step = None
    best_online_agent_perf = None
    best_online_agent_filepath = f"{project_dir}/ckpts/best_online_agent"
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(start_step, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_online_training:
            action = env.action_space.sample()
        else:
            action = agent_online.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            decoder = {"r": "return", "l": "length", "t": "time"}
            save_log(summary_writer, info["episode"], i, f"training-online{FLAGS.online_stop_step//1000}k", use_wandb=FLAGS.wandb, decoder=decoder)

        if i >= FLAGS.start_online_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent_online.update(batch)

            if (i % FLAGS.log_interval == 0) or (i == max_steps):
                save_log(summary_writer, update_info, i, f"training-online{FLAGS.online_stop_step//1000}k", use_wandb=FLAGS.wandb)

        if i in online_eval_steps: #(i % FLAGS.eval_interval == 0) or (i == max_steps):
            eval_info = evaluate(agent_online, eval_env, num_episodes=eval_episodes_per_evaluation)
            agent_perf = eval_info['return']
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            online_eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_perf}\n")
            online_eval_file.flush()
            save_log(summary_writer, eval_info, i, f"evaluation-online{FLAGS.online_stop_step//1000}k", use_wandb=FLAGS.wandb)

            if best_online_agent_perf is None or best_online_agent_perf < agent_perf:
                best_online_agent_step = i
                best_online_agent_perf = agent_perf
                save_agent(orbax_checkpointer, agent_online, i, best_online_agent_filepath)

            # log the best model performance so far
            save_log(summary_writer, {"best_ckpt_return":best_online_agent_perf}, i, "evaluation", use_wandb=FLAGS.wandb)

    online_eval_file.write(f"**Best Policy**\t{best_online_agent_step}\t{best_online_agent_perf}\n")
    online_eval_file.close()


    # Phase 2: Offline RL Phase:
    ## prepare the "best" online agent for the data collection in the 1st offline procedure
    if FLAGS.apply_best_pooling:
        kwargs_online = dict(FLAGS.config_online)
        agent_data_collection = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs_online)
        load_agent(orbax_checkpointer, agent_data_collection, best_online_agent_filepath)
    else: # no best pooling is applied. That is, the use the last policy of online RL learning as the data-collection policy
        agent_data_collection = agent_online

    ## prepare the dataset for the offline RL
    if FLAGS.online_budget_per_offline_trigger == 0: # the Buffer Strategy
        # use the replay buffer of the online learning as the dataset for the offline RL
        print("Online2Offline uses the online replay buffer as the dataset for the offline RL")
        experience_collection_tag=f"withbuffer{FLAGS.online_stop_step}"
        dataset_offline = replay_buffer
        consumed_budget = FLAGS.online_stop_step
        left_budget_for_online_finetunning = FLAGS.online_budget_total-FLAGS.online_stop_step 
        if FLAGS.apply_best_pooling:
            left_budget_for_online_finetunning -= (int(FLAGS.eval_budget_base * FLAGS.offline_eval_budget_ratio))
            consumed_budget += (int(FLAGS.eval_budget_base * FLAGS.offline_eval_budget_ratio))
    else: # the NewData Strategy
        # collect new trajectories as the dataset for the offline RL
        print(f"Online2Offline uses the newly collected trajectories of {FLAGS.online_budget_per_offline_trigger} steps as the dataset for the offline RL")

        eps_rng = jax.random.PRNGKey(FLAGS.seed)
        new_experience_collection_budget = FLAGS.online_budget_per_offline_trigger 
        prev_consumed_budget = FLAGS.online_stop_step
        consumed_budget = prev_consumed_budget + new_experience_collection_budget
        left_budget_for_online_finetunning = FLAGS.online_budget_total-FLAGS.online_stop_step-new_experience_collection_budget
        if FLAGS.apply_best_pooling:
            left_budget_for_online_finetunning -= (int(FLAGS.eval_budget_base * FLAGS.online_eval_budget_ratio) + int(FLAGS.eval_budget_base * FLAGS.offline_eval_budget_ratio))
            consumed_budget += (int(FLAGS.eval_budget_base * FLAGS.online_eval_budget_ratio) + int(FLAGS.eval_budget_base * FLAGS.offline_eval_budget_ratio))
            prev_consumed_budget += int(FLAGS.eval_budget_base * FLAGS.online_eval_budget_ratio)

        new_experiences_buffer = ReplayBuffer(
            env.observation_space, env.action_space, FLAGS.online_budget_total,
        )
        new_experiences_buffer.seed(FLAGS.seed)

        # collect new experiences using the reference agent
        observation, done = env.reset(), False
        for i in tqdm.tqdm(
            range(1, new_experience_collection_budget + 1), smoothing=0.1, disable=not FLAGS.tqdm
        ):
            if FLAGS.eps_exploration == 0:
                action = agent_data_collection.eval_actions(observation)
            else: # >0
                eps_rng, key = jax.random.split(eps_rng)
                if jax.random.uniform(key) < FLAGS.eps_exploration:
                    action = env.action_space.sample() # random action
                else:
                    action = agent_data_collection.eval_actions(observation)

            next_observation, reward, done, info = env.step(action)

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            new_experiences_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )
            observation = next_observation

            if done:
                observation, done = env.reset(), False

        experience_collection_tag=f"from{prev_consumed_budget}to{consumed_budget}"
        dataset_offline = new_experiences_buffer

    if FLAGS.save_offline_dataset:
        dataset_offline_fp = os.path.join(project_dir, "dataset_for_offline_learning.h5py")
        dataset_offline.save_dataset_h5py(dataset_offline_fp)
 

    ## Apply the offline RL to learn policies
    offline_eval_filepath = f"{project_dir}/eval_ave_episode_return_offline{experience_collection_tag}.txt"
    with open(offline_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tStep\tReturn\n")

    offline_eval_file = open(offline_eval_filepath, "a")

    best_offline_agent_step = None
    best_offline_agent_perf = None
    best_offline_agent_filepath = f"{project_dir}/ckpts/best_offline_agent_{experience_collection_tag}"

    # create the iql agent and initialize the orbax checkpointer for saving the agent periodically
    kwargs_offline = dict(FLAGS.config_offline.model_config)
    if FLAGS.use_fixed_lr_for_both_offline_and_finetunning:
        kwargs_offline.pop("cosine_decay", False)
        kwargs_offline["decay_steps"] = None
    elif kwargs_offline.pop("cosine_decay", False):
        kwargs_offline["decay_steps"] = FLAGS.max_offline_gradient_steps

    agent_offline = globals()[FLAGS.config_offline.model_constructor](
        FLAGS.seed, eval_env.observation_space.sample(), eval_env.action_space.sample(), **kwargs_offline
    )

   
    for i in tqdm.tqdm(
        range(1, FLAGS.max_offline_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = dataset_offline.sample(FLAGS.batch_size)
        info = agent_offline.update(batch)

        if i % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            save_log(summary_writer, info, i, f"training-offline-{experience_collection_tag}", use_wandb=FLAGS.wandb)

        if i in offline_eval_steps: #i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent_offline, eval_env, num_episodes=eval_episodes_per_evaluation)
            # normalize the return: use the Reference MAX and MIN returns stored in
            # the registred environments defined in D4RL.
            # ToDO: use the MAX and MIN evaluation returns from online learning with the same seed
            if FLAGS.normalize_eval_return:
                eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0

            agent_perf = eval_info['return']
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            offline_eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_perf}\n")
            offline_eval_file.flush()

            if best_offline_agent_perf is None or best_offline_agent_perf < agent_perf:
                best_offline_agent_step = i
                best_offline_agent_perf = agent_perf
                save_agent(orbax_checkpointer, agent_offline, i, best_offline_agent_filepath)

            save_log(summary_writer, eval_info, i, f"evaluation-offline-{experience_collection_tag}", use_wandb=FLAGS.wandb)
    # log the performance of the best offline-optimized agent
    save_log(summary_writer, {"best_ckpt_return":best_offline_agent_perf}, consumed_budget, "evaluation", use_wandb=FLAGS.wandb)
    expr_now = datetime.now()
    expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
    step = consumed_budget
    offline_eval_file.write(f"**Best Policy**\t{best_offline_agent_step}\t{best_offline_agent_perf}\n")
    offline_eval_file.flush()
    offline_eval_file.close()


    # Phase 3: online finetunning
    ## Prepare the offline optimized policy before online fine-tuning
    if FLAGS.apply_best_pooling:
        kwargs_offline = dict(FLAGS.config_offline.model_config)
        if FLAGS.use_fixed_lr_for_both_offline_and_finetunning:
            kwargs_offline.pop("cosine_decay", False)
            kwargs_offline["decay_steps"] = None
        elif kwargs_offline.pop("cosine_decay", False):
            kwargs_offline["decay_steps"] = FLAGS.max_offline_gradient_steps
        agent_finetunning = globals()[FLAGS.config_offline.model_constructor](
            FLAGS.seed, eval_env.observation_space.sample(), eval_env.action_space.sample(), **kwargs_offline
        )
        load_IQL_agent(orbax_checkpointer, agent_finetunning, best_offline_agent_filepath)
    else:
        agent_finetunning = agent_offline

    finetunning_budget = left_budget_for_online_finetunning
    ## Update the optimizer accordingly
    if not FLAGS.use_fixed_lr_for_both_offline_and_finetunning:
        actor_lr = kwargs_offline["actor_lr"] 
        if FLAGS.use_cosine_decay_for_finetunning:
            min_finetunning_lr = actor_lr/2.0
            update_iql_actor_optimizer_with_new_cosine_scheduler_for_finetunning(agent_finetunning, actor_lr, min_finetunning_lr, finetunning_budget)
        else:
            update_iql_actor_optimizer_with_fixed_lr_for_finetunning(agent_finetunning, actor_lr)
        

    ## Run online finetunning
    ### Initialize the replay buffer for fine-tuning with the offline dataset, as this script employs IQL for fine-tuning, consistent with the approach used in the IQL paper to initialize the replay buffer.
    finetunning_replay_buffer = dataset_offline
    from_step = FLAGS.online_budget_total-finetunning_budget
    finetunning_tag = f"from{from_step//1000}kto{FLAGS.online_budget_total//1000}k"
    finetunning_eval_filepath = f"{project_dir}/eval_ave_episode_return_finetunning{finetunning_tag}.txt"
    with open(finetunning_eval_filepath, "w") as f:
        f.write(f"Experiment Time\tStep\tReturn\n")

    finetunning_eval_file = open(finetunning_eval_filepath, "a")

    start_step = from_step+1
    max_steps = FLAGS.online_budget_total
    best_finetunning_agent_step = None
    best_finetunning_agent_perf = None
    best_finetunning_agent_filepath = f"{project_dir}/ckpts/best_finetunned_agent"
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(start_step, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        action = agent_finetunning.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        finetunning_replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            decoder = {"r": "return", "l": "length", "t": "time"}
            save_log(summary_writer, info["episode"], i, f"training-finetunning{finetunning_tag}", use_wandb=FLAGS.wandb, decoder=decoder)

        batch = finetunning_replay_buffer.sample(FLAGS.batch_size)
        update_info = agent_finetunning.update(batch)

        if ((i-from_step) % FLAGS.log_interval == 0) or (i == max_steps):
            save_log(summary_writer, update_info, i, f"training-finetunning{finetunning_tag}", use_wandb=FLAGS.wandb)

        # no best pooling is applied in online finetunning process and return the last policy as the output policy
        if ((i-from_step) % FLAGS.eval_interval == 0) or (i == max_steps):
            eval_info = evaluate(agent_finetunning, eval_env, num_episodes=FLAGS.eval_episodes)
            agent_perf = eval_info['return']
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            finetunning_eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_perf}\n")
            finetunning_eval_file.flush()
            save_log(summary_writer, eval_info, i, f"evaluation-finetunning{finetunning_tag}", use_wandb=FLAGS.wandb)

            if best_finetunning_agent_perf is None or best_finetunning_agent_perf < agent_perf:
                best_finetunning_agent_step = i
                best_finetunning_agent_perf = agent_perf
                save_agent(orbax_checkpointer, agent_finetunning, i, best_finetunning_agent_filepath)

            # log the best model performance so far
            save_log(summary_writer, {"best_ckpt_return":best_finetunning_agent_perf}, i, "evaluation", use_wandb=FLAGS.wandb)

    finetunning_eval_file.write(f"**Best Policy**\t{best_finetunning_agent_step}\t{best_finetunning_agent_perf}\n")
    finetunning_eval_file.close()


if __name__ == "__main__":
    app.run(main)
