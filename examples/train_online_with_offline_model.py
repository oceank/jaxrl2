#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
import json
import gym
import tqdm
import wandb
import jax
import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import SACLearner, IQLLearner
from jaxrl2.data import ReplayBuffer, Dataset, merge_dataset_dicts
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

def cal_online_agent_sel_prob(
    agent_online_perf:float, agent_offline_perf:float, agent_random_perf:float,
    agent_online_eval_perfs, window_size
    )->float:
    """_summary_

    Args:
        agent_online_perf (float): the performance of the online agent
        agent_offline_perf (float): the performance of the offline agent
        agent_random_perf (float): the performance of the random agent

    Returns:
        online_agent_sel_prob (float): a probability of selecting online agent to act
    """
    online_agent_better_perf_steady = False
    if len(agent_online_eval_perfs) >= window_size:
        online_agent_better_perf_steady = (agent_online_perf > agent_offline_perf)
        last_few_evals = agent_online_eval_perfs[-window_size:]
        last_few_evals_ave = sum(last_few_evals)/window_size
        online_agent_better_perf_steady = online_agent_better_perf_steady and (last_few_evals_ave>agent_offline_perf)
        
    if online_agent_better_perf_steady:
        online_agent_sel_prob = 1.0
    else:
        online_perf_diff = agent_online_perf - agent_random_perf
        offline_perf_diff = agent_offline_perf - agent_random_perf
        online_agent_sel_prob = 1.0
        if offline_perf_diff <= 0:
            online_agent_sel_prob = 1.0
        elif online_perf_diff <= 0:
            online_agent_sel_prob = 0.0
        else:
            online_agent_sel_prob = agent_online_perf/(agent_online_perf+agent_offline_perf)
            if online_agent_sel_prob < 0.1:
                online_agent_sel_prob = 0.1
    return online_agent_sel_prob

Training_Testing_Seed_Gap = 10000
FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")

flags.DEFINE_string("offline_learning_dataset_folder_name", "", "The name of the folder containing the offline learning dataset")
flags.DEFINE_string("loaded_online_model_name", "", "The name of the loaded online model, including the experiment name and ckpt that are separated by a colon")
flags.DEFINE_string("loaded_offline_model_name", "", "The name of the loaded offline model, including the experiment name and ckpt that are separated by a colon")
flags.DEFINE_string("experience_collection_mode", "equal", "The mode of experience collection for online learning. Default is equal, indicating online-learned model and offline-learned model equally collecting new experiences. Currently, its value can be one of  ['equal', 'weighted', 'online']")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("online_start_step", int(1), "Number of the start step of online training.")
flags.DEFINE_integer("max_steps", int(1e6), "The total budget of steps for interacting with the env.")
flags.DEFINE_integer(
    "initial_collection_steps", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("offline_procedure_step", int(800000), "The step when the online learning stops and the offline procedure kicks in.")
flags.DEFINE_integer("window_size", 3, "The window size for calculating the average performance of the online agent in the last few evaluations.")
flags.DEFINE_boolean("save_online_replay_buffer", False, "Save the online replay buffer.")
flags.DEFINE_boolean("save_best", True, "Save the best model.")
flags.DEFINE_boolean("save_ckpt", False, "Save the checkpoints.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", False, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
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

# Path 1: purely online (on)
# Path 2: online2offline (on2of)
# path 3: online2offline2online (on2of2on)
#
# Seeding:
# 1. Seed the environment and the action space
# 2. Seed the agent
# 3. Seed the replay buffer

def main(_):
    # create the project directory
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    project_name = f"{FLAGS.env_name}_seed{FLAGS.seed}_on_sac"
    
    online_initial_model_tag = "randomInit"
    online_ckpt_step = FLAGS.offline_procedure_step

    if FLAGS.loaded_online_model_name != "":
        pieces = FLAGS.loaded_online_model_name.split(":")
        online_model_experiment_name = pieces[0]
        online_ckpt_name = pieces[1]
        online_initial_model_tag = online_ckpt_name[:4] + online_ckpt_name[5:] + "Init"
        if "best_ckpts" in online_ckpt_name:
            online_ckpt_step = int(online_ckpt_name[11:])
            topk="top1"
            loading_online_agent_filepath = os.path.join(FLAGS.save_dir, online_model_experiment_name, "ckpts", online_ckpt_name, topk)
            import csv
            with open(os.path.join(FLAGS.save_dir, online_model_experiment_name, "ckpts", online_ckpt_name, "top_n_performace.csv"), "r") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if row[0] == "Name":
                        continue
                    if row[0] == topk:
                        online_ckpt_step = int(row[2])
                        FLAGS.offline_procedure_step = online_ckpt_step
                        break
        else:
            online_ckpt_step = int(online_ckpt_name[5:]) # if online_ckpt_name is not "best_ckpt", it is like "ckpt_1000000
            loading_online_agent_filepath = os.path.join(FLAGS.save_dir, online_model_experiment_name, "ckpts", online_ckpt_name)
        FLAGS.offline_learning_dataset_folder_name = online_model_experiment_name
        assert FLAGS.online_start_step == (1+FLAGS.offline_procedure_step)


    pieces = FLAGS.loaded_offline_model_name.split(":")
    offline_model_experiment_name = pieces[0]
    behavior_policy_dataset_tag = offline_model_experiment_name.split("_")[4]
    offline_model_ckpt = pieces[1] # e.g., "ckpt_1000000", "best_ckpt"
    # FLAGS.offline_learning_dataset_filename: e.g., new_20000_experiences_by_80000_top3 indicates a collection of 20k transitions using the top3 policies by ckpt_80000
    behavior_policy_dataset_filename = pieces[2]
    offline_dataset_size = int(behavior_policy_dataset_filename.split("_")[1])
    loading_offline_agent_filepath = os.path.join(FLAGS.save_dir, offline_model_experiment_name, "ckpts", offline_model_ckpt)
    # remove '_' in loaded_model_ckpt for the consistency of experiment name
    offline_ckpt_simplified_name = offline_model_ckpt[:4] + offline_model_ckpt[5:]
    
    project_name += f"_{online_initial_model_tag}_{behavior_policy_dataset_tag}-{offline_ckpt_simplified_name}"
    project_name += f"_{expr_time_str}"
    project_dir = os.path.join(FLAGS.save_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # save the machine's nvidia-smi output
    save_machine_info(f"{project_dir}/machine_info.txt")

    # save configuration to file
    flags_dict = flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{project_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict[_[0]].items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4)

    # Initialize the logger for the experiment
    if FLAGS.wandb:
        summary_writer = wandb.init(project=project_name)
        summary_writer.config.update(FLAGS)
    else:
        summary_writer = SummaryWriter(project_dir, write_to_disk=True)

    # create the environments
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + Training_Testing_Seed_Gap)

    # Initialize the orbax checkpointer for loading prior agents and saving the online agent periodically
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    kwargs_online = dict(FLAGS.config)
    agent_online = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs_online)
    if FLAGS.loaded_online_model_name != "":
        load_agent(orbax_checkpointer, agent_online, loading_online_agent_filepath)
    eval_info_agent_online = evaluate(agent_online, eval_env, num_episodes=FLAGS.eval_episodes)
    agent_online_perf = eval_info_agent_online['return']

    ## save the initial performance of the online agent and the offline agent
    eval_filepath = f"{project_dir}/eval_ave_episode_return.txt"
    with open(eval_filepath, "w") as f:
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = FLAGS.online_start_step-1
        f.write(f"Experiment Time\tStep\tReturn\n")
        f.write(f"{expr_time_now_str}\t{step}\t{agent_online_perf}\n")
    eval_file = open(eval_filepath, "a")
    save_log(summary_writer, eval_info_agent_online, FLAGS.online_start_step-1, "evaluation", use_wandb=FLAGS.wandb)
    ## save the initial best agent
    if FLAGS.save_best:
        best_ckpt_filepath = f"{project_dir}/ckpts/best_ckpt"
        best_ckpt_performance = {"best_ckpt_return":agent_online_perf, "best_ckpt_step":(FLAGS.online_start_step-1)}
        save_agent(orbax_checkpointer, agent_online, FLAGS.online_start_step-1, best_ckpt_filepath)
    ## save the checkpoint at step, FLAGS.online_start_step-1: the initial agent
    if FLAGS.save_ckpt:
        ckpt_filepath = f"{project_dir}/ckpts/ckpt_{FLAGS.online_start_step-1}"
        save_agent(orbax_checkpointer, agent_online, FLAGS.online_start_step-1, ckpt_filepath)

    # Initialize the replay buffer with the previously saved one, and add the offline dataset to the replay buffer as necessary
    replay_buffer_online = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer_online.seed(FLAGS.seed)
    ## loaded the replay buffer at the time step, online_ckpt_step, if FLAGS.loaded_online_model_name is not ""
    if FLAGS.loaded_online_model_name:
        prev_replay_buffer_filepath = os.path.join(FLAGS.save_dir, online_model_experiment_name, "final_replay_buffer.h5py")
        prev_buffer_dict, metadata_online = ReplayBuffer.load_dataset_h5py(prev_replay_buffer_filepath)
        replay_buffer_online.insert_chunk(prev_buffer_dict, list(range(0, online_ckpt_step, 1)))

    ## load the offline dataset as a source of replay buffer
    offline_dataset_filepath = os.path.join(FLAGS.save_dir, FLAGS.offline_learning_dataset_folder_name, f"{behavior_policy_dataset_filename}.h5py")        
    offline_dataset, metadata_offline = ReplayBuffer.load_dataset_h5py(offline_dataset_filepath)
    replay_buffer_offline = Dataset(dataset_dict=offline_dataset, seed=FLAGS.seed)
        
    # Initialize the online learning procedure
    observation, done = env.reset(), False
    online_start_step = FLAGS.online_start_step
    max_steps = FLAGS.offline_procedure_step

    for i in tqdm.tqdm(
        range(online_start_step, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        start_training = len(replay_buffer_online) >= FLAGS.initial_collection_steps
        if not start_training:
            action = env.action_space.sample()
        else:
            action = agent_online.sample_actions(observation)

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
            batch = replay_buffer_online.sample(FLAGS.batch_size)
            update_info = agent_online.update(batch)

            if (i % FLAGS.log_interval == 0) or (i == max_steps):
                save_log(summary_writer, update_info, i, "training", use_wandb=FLAGS.wandb)

        if (i % FLAGS.eval_interval == 0) or (i == max_steps):
            eval_info_agent_online = evaluate(agent_online, eval_env, num_episodes=FLAGS.eval_episodes)
            agent_online_perf = eval_info_agent_online['return']
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_online_perf}\n")
            eval_file.flush()
            save_log(summary_writer, eval_info_agent_online, i, "evaluation", use_wandb=FLAGS.wandb)

            ### save the current best agent
            if FLAGS.save_best and best_ckpt_performance["best_ckpt_return"] < agent_online_perf:
                best_ckpt_performance["best_ckpt_return"] = agent_online_perf
                best_ckpt_performance["best_ckpt_step"] = i
                save_agent(orbax_checkpointer, agent_online, i, best_ckpt_filepath)
                save_log(summary_writer, best_ckpt_performance, i, "evaluation", use_wandb=FLAGS.wandb)
            #### save the checkpoint at step i
            if FLAGS.save_ckpt and ((i<1e5 and i%1e4==0) or (i % FLAGS.ckpt_interval == 0)):
                ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
                save_agent(orbax_checkpointer, agent_online, i, ckpt_filepath)


    # Offline procedure
    ## The current code assumes the offline proceudred has been done in previous experiments.
    ## ToDo: It needs to be update to incorporate the offline procedure in the current experiment.
    ## Load the offline-learned agent
    kwargs_offline = dict(FLAGS.config_offline.model_config)
    if kwargs_offline.pop("cosine_decay", False):
        kwargs_offline["decay_steps"] = FLAGS.max_steps # the number of gradient steps for offline RL
    agent_offline = globals()[FLAGS.config_offline.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs_offline
    )
    load_agent(orbax_checkpointer, agent_offline, loading_offline_agent_filepath)
    eval_info_agent_offline = evaluate(agent_offline, eval_env, num_episodes=FLAGS.eval_episodes)
    agent_offline_perf = eval_info_agent_offline['return']
    step_budget_for_offline_dataset_collection = offline_dataset_size

    # Continue online learning
    online_start_step = FLAGS.offline_procedure_step + step_budget_for_offline_dataset_collection + 1
    save_log(summary_writer, eval_info_agent_online, online_start_step-1, "evaluation", use_wandb=FLAGS.wandb)
    max_steps = FLAGS.max_steps
    pol_sel_key = jax.random.PRNGKey(FLAGS.seed)
    num_gradient_steps = max_steps - online_start_step + 1
    pol_sel_key, subkey = jax.random.split(pol_sel_key)
    agent_sel_probs = jax.random.uniform(subkey, (num_gradient_steps,))
    eval_info_agent_random = evaluate(agent_online, eval_env, num_episodes=FLAGS.eval_episodes, random_agent=True)
    agent_random_perf = eval_info_agent_random["return"]
    start_training = True # start training the online agent since the online buffer
    agent_online_eval_perfs = [agent_online_perf]
    window_size = FLAGS.window_size # 3, 5
    online_agent_better_perf_steady = False
    for i in tqdm.tqdm(
        range(online_start_step, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        # In the current implementation, when starting the continual online learning,
        # the online buffer will have a number of transitions more than FLAGS.initial_collection_steps.
        # Leave this checking for a possible sitatuion where the online buffer is empty
        # at the start of the continual online learning.
        start_training = len(replay_buffer_online) >= FLAGS.initial_collection_steps
    
        ## based on the value of FLAGS.experience_collection_mode,
        ## decide using either online agent or offline agent to collect a new step
        if FLAGS.experience_collection_mode == "equal":
            online_agent_sel_prob = 0.5
        elif FLAGS.experience_collection_mode == "weighted":
            if not online_agent_better_perf_steady:
                online_agent_sel_prob = cal_online_agent_sel_prob(
                    agent_online_perf, agent_offline_perf, agent_random_perf,
                    agent_online_eval_perfs, window_size)
                if online_agent_sel_prob == 1.0:
                    online_agent_better_perf_steady = True
            else:
                online_agent_sel_prob = 1.0
        elif FLAGS.experience_collection_mode == "online":
            online_agent_sel_prob = 1.0
        else:
            raise ValueError(f"Invalid value for FLAGS.experience_collection_mode: {FLAGS.experience_collection_mode}")

        actioning_policy = 1 #"online"
        if agent_sel_probs[i-FLAGS.online_start_step] <= online_agent_sel_prob:
            action = agent_online.sample_actions(observation)
        else:
            action = agent_offline.sample_actions(observation)
            actioning_policy = -1 #"offline"

        action_info = {"online_agent_sel_prob":online_agent_sel_prob, "actioning_policy":actioning_policy}
        save_log(summary_writer, action_info, i, "training", use_wandb=FLAGS.wandb)

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
            if online_agent_better_perf_steady:
                # uniform sampling from the two buffers
                # offline_buffer_size_ratio = len(replay_buffer_offline)/(len(replay_buffer_offline) + len(replay_buffer_online))
                # batch_size_offline_buffer = int(FLAGS.batch_size*offline_buffer_size_ratio)

                # do not use the offline buffer
                batch_size_offline_buffer = 0
            else: # symetrical sampling from the two buffers
                batch_size_offline_buffer = int(FLAGS.batch_size//2) # symmetrical sampling of two buffers

            batch_size_online_buffer = FLAGS.batch_size - batch_size_offline_buffer
            batch_offline_buffer = replay_buffer_offline.sample(batch_size_offline_buffer)
            batch_online_buffer = replay_buffer_online.sample(batch_size_online_buffer)
            # merge thes two batches by stacking them
            batch = merge_dataset_dicts(batch_offline_buffer, batch_online_buffer)
            update_info = agent_online.update(batch)

            if (i % FLAGS.log_interval == 0) or (i == max_steps):
                save_log(summary_writer, update_info, i, "training", use_wandb=FLAGS.wandb)

        if (i % FLAGS.eval_interval == 0) or (i == max_steps):
            eval_info_agent_online = evaluate(agent_online, eval_env, num_episodes=FLAGS.eval_episodes)
            agent_online_perf = eval_info_agent_online['return']
            agent_online_eval_perfs.append(agent_online_perf)
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_online_perf}\n")
            eval_file.flush()
            save_log(summary_writer, eval_info_agent_online, i, "evaluation", use_wandb=FLAGS.wandb)

            ### save the current best agent
            if FLAGS.save_best and best_ckpt_performance["best_ckpt_return"] < agent_online_perf:
                best_ckpt_performance["best_ckpt_return"] = agent_online_perf
                best_ckpt_performance["best_ckpt_step"] = i
                save_agent(orbax_checkpointer, agent_online, i, best_ckpt_filepath)
                save_log(summary_writer, best_ckpt_performance, i, "evaluation", use_wandb=FLAGS.wandb)
            #### save the checkpoint at step i
            if FLAGS.save_ckpt and ((i<1e5 and i%1e4==0) or (i % FLAGS.ckpt_interval == 0)):
                ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
                save_agent(orbax_checkpointer, agent_online, i, ckpt_filepath)

    # save the final checkpoint
    if not FLAGS.save_ckpt:
        ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
        save_agent(orbax_checkpointer, agent_online, i, ckpt_filepath)

    if FLAGS.save_best:
        eval_file.write(f"**Best Policy**\t{best_ckpt_performance['best_ckpt_step']}\t{best_ckpt_performance['best_ckpt_return']}\n")
    eval_file.close()

    if not FLAGS.wandb: # close the tensorboard writer. wandb will close it automatically
        summary_writer.close()

    if FLAGS.save_online_replay_buffer:
        replay_buffer_metadata = {}
        replay_buffer_metadata["max_steps"] = FLAGS.max_steps
        replay_buffer_metadata["initial_collection_steps"] = FLAGS.initial_collection_steps
        replay_buffer_metadata["seed"] = FLAGS.seed
        replay_buffer_metadata["env_name"] = FLAGS.env_name
        replay_buffer_metadata["expr_time_str"] = expr_time_str
        replay_buffer_metadata["eval_interval"] = FLAGS.eval_interval
        replay_buffer_metadata["algorithm"] = "SAC"
        replay_buffer_filepath = f"{project_dir}/final_replay_buffer.h5py"
        replay_buffer_online.save_dataset_h5py(replay_buffer_filepath, metadata=replay_buffer_metadata)

if __name__ == "__main__":
    app.run(main)
