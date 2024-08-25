#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
import json
import csv
import gym
import tqdm
import wandb
import numpy as np
import jax
import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict

from jaxrl2.agents import SACLearner, IQLLearner, SACBasedPEXLearner, IQLBasedPEXLearner
from jaxrl2.data import ReplayBuffer, Dataset, merge_dataset_dicts
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_load_agent import save_agent, load_agent
from jaxrl2.utils.save_expr_log import save_log

from tensorboardX import SummaryWriter
from datetime import datetime


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


Training_Testing_Seed_Gap = 10000
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
# offline dataset filename: dataset_for_offline_learning.h5py
# offline-optimized ckpt name pattern: ckpts/best_offline_agent_*
flags.DEFINE_string("offline_dataset_and_optimized_agent_dir", "", "The folder path for storing the dataset for offline learning and the offline-optimized agent")


flags.DEFINE_string("experience_collection_mode", "equal", "The mode of experience collection for online learning. Default is equal, indicating online-learned model and offline-learned model equally collecting new experiences. Currently, its value can be one of  ['equal', 'weighted', 'online']")
flags.DEFINE_string("offline_procedure_data_tag", "New", "The type of data used in the offline procedure. The default value is 'New', indicating only using the newly collected data for offline learning. Another value is 'All', indicating use the all collected data so far (including the previous online replay buffer)")

flags.DEFINE_integer("online_stop_step", int(800000), "The step when the online learning stops and the offline procedure kicks in.")
flags.DEFINE_integer("data_collection_budget_per_offline_procedure", int(200000), "The total number of interactions for online data collection for offline procedure.")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")

flags.DEFINE_integer("total_budget", int(1e6), "The total budget of steps for interacting with the env.")
flags.DEFINE_integer(
    "initial_collection_steps", int(1e4), "Number of training steps to start training."
)

flags.DEFINE_integer("consumed_steps_by_best_pooling", 0, "Number of steps used by best pooling for evaluating policies in previous procedures")
flags.DEFINE_float("data_colllection_exploration_eps", 0.0, "The probability of applying random action during data collection for the offline procedure")

flags.DEFINE_boolean("transfer_critic", True, "Transfer the critic and value function of the offline-learned agent to initialize the online agent")
flags.DEFINE_boolean("iql_use_fixed_LR", True, "The IQL online agent uses a fixed learning rate for finetunning.")
flags.DEFINE_boolean("save_online_replay_buffer", False, "Save the online replay buffer.")
flags.DEFINE_boolean("save_best", True, "Save the best model.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
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

def retrieve_fp(check_dir, filename_substr):
    filename_list = os.listdir(check_dir)
    filename = [fn for fn in filename_list if filename_substr in fn][0]
    filepath = os.path.join(check_dir, filename)
    return filepath


# Seeding:
# 1. Seed the environment and the action space
# 2. Seed the agent
# 3. Seed the replay buffer

def main(_):
    pex_backbone = "IQL"
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{FLAGS.env_name}_seed{FLAGS.seed}_con_PEXb{pex_backbone}"
    offline_learning_dataset_tag = f"OfflineUse{FLAGS.offline_procedure_data_tag}Data{FLAGS.online_stop_step//1000}kExplorationEPS{str(FLAGS.data_colllection_exploration_eps)}" if FLAGS.data_collection_budget_per_offline_procedure else "OfflineUseBuffer" 

    experiment_name += f"_{offline_learning_dataset_tag}"

    consumed_steps = FLAGS.online_stop_step + FLAGS.data_collection_budget_per_offline_procedure + FLAGS.consumed_steps_by_best_pooling
    left_budget = FLAGS.total_budget - consumed_steps
    online_start_step = 1 + consumed_steps 

    # the name of this experiment
    #   BPBF: BestPoolingBeforeFinetunning
    #   NBPBF:NoBestPoolingBeforeFinetunning
    experiment_name += ("_NoBPBF" if FLAGS.consumed_steps_by_best_pooling==0 else "_BPBF")
    experiment_name += ("_IQLUseFixedLR" if FLAGS.iql_use_fixed_LR else "_IQLUseCosineDecayLR")
    experiment_name += f"_{expr_time_str}"
    experiment_result_dir = os.path.join(FLAGS.save_dir, experiment_name)
    os.makedirs(experiment_result_dir, exist_ok=True)

    # save the machine's nvidia-smi output
    save_machine_info(f"{experiment_result_dir}/machine_info.txt")

    # save configuration to file
    flags_dict = flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{experiment_result_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict[_[0]].items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4)

    # Initialize the logger for the experiment
    if FLAGS.wandb:
        summary_writer = wandb.init(project=experiment_name)
        summary_writer.config.update(FLAGS)
    else:
        summary_writer = SummaryWriter(experiment_result_dir, write_to_disk=True)

    # create the environments
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + Training_Testing_Seed_Gap)

    # PEX Agent Initialization
    ## Initialize the policy set
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    kwargs_offline = dict(FLAGS.config_offline.model_config)
    if FLAGS.iql_use_fixed_LR:
        kwargs_offline.pop("cosine_decay", False)
        kwargs_offline["decay_steps"] = None
    elif kwargs_offline.pop("cosine_decay", False):
        # the left budget of steps for finetunning the IQL agent
        # the offline-optimized IQL agent will not update
        kwargs_offline["decay_steps"] = left_budget
    agent_offline = globals()[FLAGS.config_offline.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs_offline
    )
    agent_online = globals()[FLAGS.config_offline.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs_offline
    )
    ## load the offline-optimized IQL agent
    offline_optimized_agent_fp = retrieve_fp(
        os.path.join(FLAGS.offline_dataset_and_optimized_agent_dir, "ckpts"), "best_offline_agent_")
    load_agent(orbax_checkpointer, agent_offline, offline_optimized_agent_fp)
    ## copy the critic and the learned value function from the offline-optimized agent to the finetunning agent
    transfer_critic = FLAGS.transfer_critic # True
    ## Ture: copy the target critic from offline-optimized agent to the finetunning agent
    ## False: (after applying the effect of 'transfer_critic'), copy the critic of the finetunning agent to the target critic
    copy_to_target = False
    ## According to the Appendix A.7 of the PEX paper, PEX follows to use the value of the inverse temperature in IQL paper
    ## 'inv_temperature' will be used in the policy selection step in PEX
    inv_temperature = kwargs_offline["A_scaling"]
    agent_pex = IQLBasedPEXLearner(FLAGS.seed, agent_online, agent_offline, inv_temperature, transfer_critic, copy_to_target)
    
    # Initial Performance of the PEX Agent
    eval_info = evaluate(agent_pex, eval_env, num_episodes=FLAGS.eval_episodes)
    agent_pex_perf = eval_info['return']
    ## the performance of the offline-optimized policy by best pooling
    agent_offline_perf = 0.0
    eval_rts_offline_fp = retrieve_fp(FLAGS.offline_dataset_and_optimized_agent_dir, "eval_ave_episode_return_offline")
    with open(eval_rts_offline_fp, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if "Best Policy" in row[0]:
                agent_offline_perf = float(row[2])
                break
            else:
                continue
    ## save the initial performance of the PEX agent
    eval_filepath = f"{experiment_result_dir}/eval_ave_episode_return.txt"
    with open(eval_filepath, "w") as f:
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = online_start_step-1
        f.write(f"Experiment Time\tStep\tReturn\n")
        #f.write(f"{expr_time_now_str}\t{consumed_steps}\t{agent_offline_perf}\n")
        #f.write(f"{expr_time_now_str}\t{step}\t{agent_pex_perf}\n")
    eval_file = open(eval_filepath, "a")
    save_log(summary_writer, eval_info, online_start_step-1, "evaluation", use_wandb=FLAGS.wandb)

    ## save the initial best agent
    if FLAGS.save_best:
        best_ckpt_filepath = f"{experiment_result_dir}/ckpts/best_finetunned_agent_by_IQLPEX"
        best_ckpt_performance = {"best_ckpt_return":agent_pex_perf, "best_ckpt_step":(online_start_step-1)}
        save_agent(orbax_checkpointer, agent_pex, online_start_step-1, best_ckpt_filepath)


    # Online and Offline Replay Buffer
    offline_learning_dataset_fp = os.path.join(FLAGS.offline_dataset_and_optimized_agent_dir, "dataset_for_offline_learning.h5py")
    ## Initialize the replay buffer with the previously saved one, and add the offline dataset to the replay buffer as necessary
    replay_buffer_online = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.total_budget
    )
    replay_buffer_online.seed(FLAGS.seed)
    ## load the offline dataset as a source of replay buffer
    offline_dataset, metadata_offline = ReplayBuffer.load_dataset_h5py(offline_learning_dataset_fp)
    replay_buffer_offline = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.total_budget
    )
    replay_buffer_offline.seed(FLAGS.seed)
    if  (FLAGS.data_collection_budget_per_offline_procedure != 0) and (FLAGS.offline_procedure_data_tag=="New"):
        num_transitions_to_select = FLAGS.data_collection_budget_per_offline_procedure
    else:
        num_transitions_to_select = FLAGS.data_collection_budget_per_offline_procedure + FLAGS.online_stop_step
    transitions_indices_to_select = np.array(range(num_transitions_to_select))
    replay_buffer_offline.insert_chunk(offline_dataset, transitions_indices_to_select)

        
    # Continue online learning

    ## Start the fine-tuning using PEX
    observation, done = env.reset(), False

    for i in tqdm.tqdm(
        range(online_start_step, FLAGS.total_budget + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        start_training = len(replay_buffer_online) >= FLAGS.initial_collection_steps
        if not start_training:
            action = env.action_space.sample()
        else: 
            action = agent_pex.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer_online.insert(
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
            save_log(summary_writer, info["episode"], i, "training", use_wandb=FLAGS.wandb, decoder=decoder)

        if start_training:
            # according to the PEX paper, the batch is sampled equally from both online and offline buffers
            batch_size_offline_buffer = int(FLAGS.batch_size//2) # symmetrical sampling of two buffers
            batch_size_online_buffer = FLAGS.batch_size - batch_size_offline_buffer
            batch_offline_buffer = replay_buffer_offline.sample(batch_size_offline_buffer)
            batch_online_buffer = replay_buffer_online.sample(batch_size_online_buffer)
            # merge thes two batches by stacking them
            batch = merge_dataset_dicts(batch_offline_buffer, batch_online_buffer)

            update_info = agent_pex.update(batch)

            if (i % FLAGS.log_interval == 0) or (i == FLAGS.total_budget):
                save_log(summary_writer, update_info, i, "training", use_wandb=FLAGS.wandb)

        if (i % FLAGS.eval_interval == 0) or (i == FLAGS.total_budget):
            eval_info = evaluate(agent_pex, eval_env, num_episodes=FLAGS.eval_episodes)
            agent_pex_perf = eval_info['return']

            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_pex_perf}\n")
            eval_file.flush()
            save_log(summary_writer, eval_info, i, "evaluation", use_wandb=FLAGS.wandb)

            ### save the current best agent
            if FLAGS.save_best and best_ckpt_performance["best_ckpt_return"] < agent_pex_perf:
                best_ckpt_performance["best_ckpt_return"] = agent_pex_perf
                best_ckpt_performance["best_ckpt_step"] = i
                save_agent(orbax_checkpointer, agent_pex, i, best_ckpt_filepath)
                save_log(summary_writer, best_ckpt_performance, i, "evaluation", use_wandb=FLAGS.wandb)


    # save the final checkpoint
    ckpt_filepath = f"{experiment_result_dir}/ckpts/last_finetunned_agent_at_step{i}"
    save_agent(orbax_checkpointer, agent_pex, i, ckpt_filepath)

    if FLAGS.save_best:
        eval_file.write(f"**Best Policy**\t{best_ckpt_performance['best_ckpt_step']}\t{best_ckpt_performance['best_ckpt_return']}\n")
    eval_file.close()

    if not FLAGS.wandb: # close the tensorboard writer. wandb will close it automatically
        summary_writer.close()

    if FLAGS.save_online_replay_buffer:
        replay_buffer_metadata = {}
        replay_buffer_metadata["total_budget"] = FLAGS.total_budget
        replay_buffer_metadata["initial_collection_steps"] = FLAGS.initial_collection_steps
        replay_buffer_metadata["seed"] = FLAGS.seed
        replay_buffer_metadata["env_name"] = FLAGS.env_name
        replay_buffer_metadata["expr_time_str"] = expr_time_str
        replay_buffer_metadata["eval_interval"] = FLAGS.eval_interval
        replay_buffer_metadata["algorithm"] = "PEX(SAC-IQL)"
        replay_buffer_filepath = f"{experiment_result_dir}/final_replay_buffer.h5py"
        replay_buffer_online.save_dataset_h5py(replay_buffer_filepath, metadata=replay_buffer_metadata)

if __name__ == "__main__":
    app.run(main)
