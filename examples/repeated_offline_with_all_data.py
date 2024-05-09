#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
import json
import csv 
import gym
import numpy as np
import tqdm
import jax
import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict
import matplotlib.pyplot as plt
from datetime import datetime

from jaxrl2.agents import SACLearner, IQLLearner
from jaxrl2.data import ReplayBuffer, plot_episode_returns
from jaxrl2.utils.save_load_agent import load_SAC_agent, load_IQL_agent, save_agent
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_expr_log import save_log

from tensorboardX import SummaryWriter

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
flags.DEFINE_string("run_name", "run", "The folder name for a saved run of policy learning.")
flags.DEFINE_string("saved_online_buffer_dirname", "buffer_dirname", "The folder name for a saved replay buffer from an online learning.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("max_offline_gradient_steps", int(1e6), "The number of gradient steps performed by the offline RL")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("online_stop_step", int(2e5), "The step when the online learning stops")
flags.DEFINE_integer("online_budget_total", int(1e6), "The total budget of steps for online interactions")
flags.DEFINE_integer("online_budget_per_offline_trigger", int(2e5), "The budget of online interactions for each offline procedure")
flags.DEFINE_float("eps_exploration", 0.1, "The probability of applying a random action instead of the action (the best action) by the reference agent")
flags.DEFINE_boolean("normalize_eval_return", False, "normalize the evaluation result")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
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

def main(_):

    # create the project directory
    now = datetime.now()
    top_n_dirname = f"best_ckpts_{FLAGS.online_stop_step}"
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    project_name = f"{FLAGS.env_name}_seed{FLAGS.seed}_rp_iql"
    project_name += f"_{top_n_dirname}_ExplorationEPS{FLAGS.eps_exploration}_AllData_{expr_time_str}"
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
    ## env for collecting new trajectories for offline procedures
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    ## eval_env is for policy evaluation during each offline procedure
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + Training_Testing_Seed_Gap)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    expr_run_dir = os.path.join(FLAGS.save_dir, FLAGS.run_name)
    sac_ckpt_fp = f"{expr_run_dir}/ckpts/{top_n_dirname}/top1"
    kwargs_online = dict(FLAGS.config_online)
    reference_agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs_online)
    print(f"===> Loading the SAC checkpoint ({sac_ckpt_fp}) as the initial reference agent")
    load_SAC_agent(orbax_checkpointer, reference_agent, sac_ckpt_fp)
    sac_ckpt_perf_fp = f"{expr_run_dir}/ckpts/{top_n_dirname}/top_n_performace.csv"
    ave_return_initial_ref_agent = 0.0 # !!!
    with open(sac_ckpt_perf_fp, mode ='r') as file:
        csvFile = csv.reader(file, delimiter="\t")
        line_num = 1
        for line in csvFile:
            if line_num == 2:
                ave_return_initial_ref_agent = float(line[1])
                break
            line_num += 1
    eval_filepath = f"{project_dir}/eval_ave_episode_return.txt"
    with open(eval_filepath, "w") as f:
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = FLAGS.online_stop_step
        f.write(f"Experiment Time\tStep\tReturn\n")
        f.write(f"{expr_time_now_str}\t{step}\t{ave_return_initial_ref_agent}\n")
    eval_file = open(eval_filepath, "a")
    save_log(summary_writer, {"best_ckpt_return":ave_return_initial_ref_agent}, FLAGS.online_stop_step, "evaluation", use_wandb=FLAGS.wandb)
   
    # buffer to save all collected trajectories for offline procedures for post analysis
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.online_budget_total
    )
    replay_buffer.seed(FLAGS.seed)
    saved_replay_buffer_filepath = os.path.join(FLAGS.save_dir, FLAGS.saved_online_buffer_dirname, "final_replay_buffer.h5py")
    saved_buffer_dict, metadata_online = ReplayBuffer.load_dataset_h5py(saved_replay_buffer_filepath)
    replay_buffer.insert_chunk(saved_buffer_dict, list(range(0, FLAGS.online_stop_step, 1)))

    # Repeat offline procedures until the entire budget of online interactions is consumed
    online_stop_step=FLAGS.online_stop_step
    online_budget_total = FLAGS.online_budget_total
    online_budget_per_offline_trigger = FLAGS.online_budget_per_offline_trigger
    consumed_budget = online_stop_step
    offline_procedure_idx = 1
    eps_rng = jax.random.PRNGKey(FLAGS.seed)
    while consumed_budget<online_budget_total:
        if (online_budget_per_offline_trigger+consumed_budget) <= online_budget_total:
            new_experience_collection_budget = online_budget_per_offline_trigger
        else:
            new_experience_collection_budget = online_budget_total - consumed_budget
        prev_consumed_budget = consumed_budget
        consumed_budget += new_experience_collection_budget
        print(f"[Offline Procedure {offline_procedure_idx}] {prev_consumed_budget} to {consumed_budget}")
        offline_procedure_idx += 1

        new_experiences_buffer = ReplayBuffer(
            env.observation_space, env.action_space, new_experience_collection_budget
        )
        new_experiences_buffer.seed(FLAGS.seed)

        # collect new experiences using the reference agent
        observation, done = env.reset(), False
        for i in tqdm.tqdm(
            range(1, new_experience_collection_budget + 1), smoothing=0.1, disable=not FLAGS.tqdm
        ):
            if FLAGS.eps_exploration < 0:
                action = reference_agent.sample_actions(observation)
            else:
                eps_rng, key = jax.random.split(eps_rng)
                if jax.random.uniform(key) < FLAGS.eps_exploration:
                    action = env.action_space.sample()
                else:
                    action = reference_agent.eval_actions(observation)

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
        plot_episode_returns(
            new_experiences_buffer,
            bin=100,
            title=f"{experience_collection_tag}",
            fig_path=f"{project_dir}/{experience_collection_tag}.png")


        replay_buffer.insert_chunk(new_experiences_buffer.dataset_dict, list(range(new_experience_collection_budget)))
        experience_collection_tag=f"from0to{consumed_budget}"
        plot_episode_returns(
            replay_buffer,
            bin=100,
            title=f"{experience_collection_tag}",
            fig_path=f"{project_dir}/{experience_collection_tag}.png")

      
        # offline RL
        # create the iql agent and initialize the orbax checkpointer for saving the agent periodically
        kwargs_offline = dict(FLAGS.config_offline.model_config)
        if kwargs_offline.pop("cosine_decay", False):
            kwargs_offline["decay_steps"] = FLAGS.max_offline_gradient_steps
        agent = globals()[FLAGS.config_offline.model_constructor](
            FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs_offline
        )
        eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
        best_ckpt_filepath = f"{project_dir}/ckpts/best_ckpt_{consumed_budget}"
        best_ckpt_performance = {"best_ckpt_return":eval_info["return"], "best_ckpt_step":0}
        save_agent(orbax_checkpointer, agent, 0, best_ckpt_filepath)

        for i in tqdm.tqdm(
            range(1, FLAGS.max_offline_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
        ):
            batch = replay_buffer.sample(FLAGS.batch_size)
            info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                info = jax.device_get(info)
                save_log(summary_writer, info, i, f"training-{prev_consumed_budget}-{consumed_budget}", use_wandb=FLAGS.wandb)

            if i % FLAGS.eval_interval == 0:
                eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
                # normalize the return: use the Reference MAX and MIN returns stored in
                # the registred environments defined in D4RL.
                # ToDO: use the MAX and MIN evaluation returns from online learning with the same seed
                if FLAGS.normalize_eval_return:
                    eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0

                # save the current best agent
                if best_ckpt_performance["best_ckpt_return"] < eval_info["return"]:
                    best_ckpt_performance["best_ckpt_return"] = eval_info["return"]
                    best_ckpt_performance["best_ckpt_step"] = i
                    save_agent(orbax_checkpointer, agent, i, best_ckpt_filepath)
                save_log(summary_writer, eval_info, i, f"evaluation-{prev_consumed_budget}-{consumed_budget}", use_wandb=FLAGS.wandb)
        # log the performance of the best offline-optimized agent
        save_log(summary_writer, {"best_ckpt_return":best_ckpt_performance["best_ckpt_return"]}, consumed_budget, "evaluation", use_wandb=FLAGS.wandb)
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = consumed_budget
        eval_file.write(f"{expr_time_now_str}\t{step}\t{best_ckpt_performance['best_ckpt_return']}\n")
        eval_file.flush()

        # update the reference agent
        reference_agent = agent
        load_IQL_agent(orbax_checkpointer, reference_agent, best_ckpt_filepath)
   

    replay_buffer_metadata = {}
    replay_buffer_metadata["policy_learning_dir"] = FLAGS.run_name
    replay_buffer_metadata["online_replay_buffer_dir"] = FLAGS.saved_online_buffer_dirname
    replay_buffer_metadata["seed"] = FLAGS.seed
    replay_buffer_metadata["env_name"] = FLAGS.env_name
    experience_collection_tag = f"new_{online_budget_total-online_stop_step}_experiences_repeated_every_{online_budget_per_offline_trigger}_from_ckpt{online_stop_step}"
    replay_buffer_filepath = f"{project_dir}/{experience_collection_tag}.h5py"
    replay_buffer.save_dataset_h5py(replay_buffer_filepath, metadata=replay_buffer_metadata)


if __name__ == "__main__":
    app.run(main)
