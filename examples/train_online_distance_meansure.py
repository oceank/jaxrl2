#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

import shutil
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
from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate, evaluate_with_disc
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
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 100000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("reward_accumulate_steps", 1, "Number of steps to accumulate the reward for sparsing the orignal reward.")
flags.DEFINE_integer("save_best_n", 1, "Save the best n models when save_best is true")
flags.DEFINE_boolean("save_best", True, "Save the best model.")
flags.DEFINE_boolean("save_ckpt", True, "Save the checkpoints.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# Seeding:
# 1. Seed the environment and the action space
# 2. Seed the agent
# 3. Seed the replay buffer

def main(_):

    top_n_checkpoints = [200000, 400000, 600000, 800000, 1000000, 1100000, 1250000, 1500000]

    online_initial_model_tag = "randomInit"

    # create the project directory
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    project_name = f"{FLAGS.env_name}-ras{FLAGS.reward_accumulate_steps}_seed{FLAGS.seed}_on_sac"
    project_name += f"_{online_initial_model_tag}_{expr_time_str}"
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
    reward_accumulate_steps=FLAGS.reward_accumulate_steps
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True, reward_accumulate_steps=reward_accumulate_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True, reward_accumulate_steps=reward_accumulate_steps)
    eval_env.seed(FLAGS.seed + Training_Testing_Seed_Gap)

    # create the replay buffer or initialized from the previous replay buffer
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    # create the agent and initialize the orbax checkpointer for saving the agent periodically
    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # evaluate the initial agent
    # evaluate_with_disc
    eval_info = evaluate_with_disc(agent, eval_env, num_episodes=FLAGS.eval_episodes)
    agent_perf = eval_info['return']
    Q_pi_disc = eval_info['rmse']
    print(f"[Step 0] agent's performance {agent_perf}, discrepancy between Q and pi {Q_pi_disc}")

    # save the performance of the initial agent
    start_step = 1
    eval_filepath = f"{project_dir}/eval_ave_episode_return.txt"
    with open(eval_filepath, "w") as f:
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = start_step-1
        f.write(f"Experiment Time\tStep\tReturn\tDisc\n")
        f.write(f"{expr_time_now_str}\t{step}\t{agent_perf}\t{Q_pi_disc}\n")

    eval_file = open(eval_filepath, "a")
    save_log(summary_writer, eval_info, start_step-1, "evaluation", use_wandb=FLAGS.wandb)
    # save the initial best agent
    if FLAGS.save_best:
        best_ckpt_dir = f"{project_dir}/ckpts/best_ckpts_{FLAGS.max_steps}"
        top1_ckpt_filepath = f"{best_ckpt_dir}/top1"
        best_ckpt_performance = {"top1":{"return":agent_perf, "step":(start_step-1), "filepath":top1_ckpt_filepath}}
        save_agent(orbax_checkpointer, agent, (start_step-1), top1_ckpt_filepath)
        for k in range(2, FLAGS.save_best_n+1, 1):
            topk = f"top{k}"
            topk_ckpt_filepath = f"{best_ckpt_dir}/{topk}"
            best_ckpt_performance[topk] = {"return": float("-inf"), "step":-1, "filepath":topk_ckpt_filepath}
            save_agent(orbax_checkpointer, agent, -k+1, topk_ckpt_filepath)

    # save the checkpoint at step 0: the initial agent
    if FLAGS.save_ckpt:
        ckpt_filepath = f"{project_dir}/ckpts/ckpt_{start_step-1}"
        save_agent(orbax_checkpointer, agent, (start_step-1), ckpt_filepath)


    observation, done = env.reset(), False
    max_steps = FLAGS.max_steps

 
    for i in tqdm.tqdm(
        range(start_step, max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
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
            save_log(summary_writer, info["episode"], i, "training", use_wandb=FLAGS.wandb, decoder=decoder)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if (i % FLAGS.log_interval == 0) or (i == max_steps):
                save_log(summary_writer, update_info, i, "training", use_wandb=FLAGS.wandb)

        if (i % FLAGS.eval_interval == 0) or (i == max_steps):
            eval_info = evaluate_with_disc(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            agent_perf = eval_info['return']
            Q_pi_disc = eval_info['rmse']
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            eval_file.write(f"{expr_time_now_str}\t{step}\t{agent_perf}\t{Q_pi_disc}\n")
            eval_file.flush()
            save_log(summary_writer, eval_info, i, "evaluation", use_wandb=FLAGS.wandb)
            print(f"[Step i] agent's performance {agent_perf}, the discrepancy between Q and pi {Q_pi_disc}")

            # save the current best agent
            # assume there best n models are saved and the current model is better than the top kth model but not (k+1)th model
            # remove the top n model in drive
            # rename the top kth model to be top (k+1)th model until k=n-1
            # save the current model as the top kth model
            if FLAGS.save_best:
                # find the first top model that performs worse than the current model
                first_top_k = 1
                while first_top_k <= FLAGS.save_best_n:
                    if best_ckpt_performance[f"top{first_top_k}"]["return"] >= agent_perf:
                        first_top_k += 1
                    else:
                        break
                if first_top_k <= FLAGS.save_best_n:
                    # Top models perform worse than the current model: 
                    #   top{first_top_k}, top{first_top_k+1}, ..., top{FLAGS.save_best_n}
                    # Totally, there are FLAGS.save_best_n-first_top_k+1) such models
                    # Move the top{FLAGS.save_best_n - 1} to top{FLAGS.save_best_n}
                    # Move the top{FLAGS.save_best_n - 2} to top{FLAGS.save_best_n - 1}
                    # ...
                    # Move the top{first_top_k} to the top{first_top_k+1}
                    # Then, save the current model to the top{first_top_k}
                    for move_iter in range(FLAGS.save_best_n-first_top_k):
                        k = FLAGS.save_best_n - 1 - move_iter
                        source = f"top{k}"
                        target = f"top{k+1}"
                        best_ckpt_performance[target]["return"] = best_ckpt_performance[source]["return"]
                        best_ckpt_performance[target]["step"] = best_ckpt_performance[source]["step"]
                        source_fp = best_ckpt_performance[source]["filepath"]
                        target_fp = best_ckpt_performance[target]["filepath"]
                        for (root, dirs, file) in os.walk(source_fp):
                            for f in file:
                                os.replace(os.path.join(source_fp, f), os.path.join(target_fp, f))

                    topk = f"top{first_top_k}"
                    best_ckpt_performance[topk]["return"] = agent_perf
                    best_ckpt_performance[topk]["step"] = i
                    save_agent(
                        orbax_checkpointer, agent, i, 
                        best_ckpt_performance[topk]["filepath"], 
                        )

                # if the current step, e.g. 80000, is one of predefined step to save top n policies,
                # save the current top n models into best_ckpts_80000 folder under the ckpts folder
                if (i in top_n_checkpoints) and (i != FLAGS.max_steps):
                    best_ckpt_dir_i = f"{project_dir}/ckpts/best_ckpts_{i}"
                    for k in range(1, FLAGS.save_best_n+1, 1):
                        topk = f"top{k}"
                        topk_ckpt_filepath = f"{best_ckpt_dir_i}/{topk}"
                        shutil.copytree(best_ckpt_performance[topk]["filepath"], topk_ckpt_filepath)
                    with open(os.path.join(best_ckpt_dir_i, "top_n_performace.csv"), "w") as f:
                        f.write(f"Name\tPerformance\tStep\n")
                        for k in range(1, FLAGS.save_best_n+1, 1):
                            topk = f"top{k}"
                            rt = best_ckpt_performance[topk]["return"]
                            sp = best_ckpt_performance[topk]["step"]
                            f.write(f"{topk}\t{rt}\t{sp}\n")
                    ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
                    save_agent(orbax_checkpointer, agent, i, ckpt_filepath)

                # log the best model performance so far
                save_log(summary_writer, {"best_ckpt_return":best_ckpt_performance["top1"]["return"]}, i, "evaluation", use_wandb=FLAGS.wandb)

            # save the checkpoint at step i
            if FLAGS.save_ckpt and (i % FLAGS.ckpt_interval == 0):
                ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
                save_agent(orbax_checkpointer, agent, i, ckpt_filepath)


    if FLAGS.save_best:
        with open(os.path.join(best_ckpt_dir, "top_n_performace.csv"), "w") as f:
            f.write(f"Name\tPerformance\tStep\n")
            for k in range(1, FLAGS.save_best_n+1, 1):
                topk = f"top{k}"
                rt = best_ckpt_performance[topk]["return"]
                sp = best_ckpt_performance[topk]["step"]
                f.write(f"{topk}\t{rt}\t{sp}\n")

        eval_file.write(f"**Best Policy**\t{best_ckpt_performance['top1']['step']}\t{best_ckpt_performance['top1']['return']}\n")
    eval_file.close()

    if not FLAGS.wandb: # close the tensorboard writer. wandb will close it automatically
        summary_writer.close()

    replay_buffer_metadata = {}
    replay_buffer_metadata["max_steps"] = FLAGS.max_steps
    replay_buffer_metadata["start_training"] = FLAGS.start_training
    replay_buffer_metadata["seed"] = FLAGS.seed
    replay_buffer_metadata["env_name"] = FLAGS.env_name
    replay_buffer_metadata["expr_time_str"] = expr_time_str
    replay_buffer_metadata["eval_interval"] = FLAGS.eval_interval
    replay_buffer_metadata["algorithm"] = "SAC"
    replay_buffer_filepath = f"{project_dir}/final_replay_buffer.h5py"
    replay_buffer.save_dataset_h5py(replay_buffer_filepath, metadata=replay_buffer_metadata)

if __name__ == "__main__":
    app.run(main)
