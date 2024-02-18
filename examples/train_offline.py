#! /usr/bin/env python
import os
# XLA GPU Deterministic Ops: https://github.com/google/jax/discussions/10674
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import subprocess
import json
import gym
import numpy as np
import jax
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from ml_collections.config_dict.config_dict import ConfigDict

from jaxrl2.agents import BCLearner, IQLLearner
from jaxrl2.data import D4RLDataset, ReplayBuffer, Dataset, plot_episode_returns
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_load_agent import save_agent
from jaxrl2.utils.save_expr_log import save_log

from tensorboardX import SummaryWriter
from datetime import datetime

def save_nvidia_smi_output(filename):
  """Saves the output of nvidia-smi to a file.

  Args:
    filename: The name of the file to save the output to.
  """

  commands = ["hostname", "echo", "nvidia-smi"]
  with open(filename, "w") as f:
      for command in commands:
        output = subprocess.check_output(command)
        f.write(output.decode("utf-8"))

self_collect_dataset_tags = {
    "RandomPolicy": ["ckpt0"],
    "ExpertPolicy": ["ckpt1000000"],
    "MediumPolicy": {
        "halfcheetah": ["ckpt100000", "ckpt200000"],
    }
}

def get_num_short_name(steps) -> str:
    if steps < 1e3:
        return f"{steps}"
    elif steps < 1e6:
        return f"{int(steps/1e3)}k"
    else:
        return f"{int(steps/1e6)}m"
def get_dataset_tag(dataset_name, env_name):
    dataset_tag = ""
    if dataset_name != "d4rl":
        raw_env_name = env_name.split("-")[0]
        ckpt_name = "".join(dataset_name.split("_")[-2:])
        for behavior_policy_tag in self_collect_dataset_tags:
            if behavior_policy_tag == "MediumPolicy":
                if ckpt_name in self_collect_dataset_tags[behavior_policy_tag][raw_env_name.lower()]:
                    dataset_tag = behavior_policy_tag+get_num_short_name(int(dataset_name.split('_')[-1]))
                    break
            else:
                if ckpt_name in self_collect_dataset_tags[behavior_policy_tag]:
                    dataset_tag = behavior_policy_tag
                    break
    else:
        dataset_tag = "d4rl"
    return dataset_tag


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("dataset_name", "d4rl", "the source name of offline dataset.")
flags.DEFINE_string("dataset_dir", None, "the path of the directory that contains the dataset (not from d4rl).")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Log wandb.")
flags.DEFINE_boolean("save_best", True, "Save the best model.")
flags.DEFINE_boolean("normalize_eval_return", True, "Normalize the average return in evaluation.")
flags.DEFINE_float("filter_percentile", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
config_flags.DEFINE_config_file(
    "config",
    "configs/offline_config.py:bc",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    # create the project directory
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    offline_algo="iql"
    dataset_names = FLAGS.dataset_name.split(",")
    dataset_tag = "".join([get_dataset_tag(dn, FLAGS.env_name) for dn in dataset_names])
    project_name = f"{FLAGS.env_name}_seed{FLAGS.seed}_off_{offline_algo}_{dataset_tag}_{expr_time_str}"
    project_dir = os.path.join(FLAGS.save_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # save the machine's nvidia-smi output
    save_nvidia_smi_output(f"{project_dir}/machine_info.txt")

    # save configuration to file
    flags_dict = flags.FLAGS.flags_by_module_dict()
    flags_dict = {k: {v.name: v.value for v in vs} for k, vs in flags_dict.items()}
    expr_config_filepath = f"{project_dir}/expr_config.json"
    expr_config_dict = {k:(v.to_dict() if isinstance(v, ConfigDict) else v) for k, v in flags_dict[_[0]].items()}
    with open(expr_config_filepath, "w") as f:
        json.dump(expr_config_dict, f, indent=4)

    if FLAGS.wandb:
        summary_writer = wandb.init(project=project_name)
        summary_writer.config.update(FLAGS)
    else:
        summary_writer = SummaryWriter(project_dir, write_to_disk=True)

    env = gym.make(FLAGS.env_name) # how does the name, "halfcheetah-expert-v2", map to the actual environment?
    env = wrap_gym(env)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)

    if FLAGS.dataset_name == "d4rl":
        dataset = D4RLDataset(env)
    else:
        assert FLAGS.dataset_dir, "Please specify the dataset directory."
        dataset = None
        for dn in dataset_names:
            dataset_path = os.path.join(FLAGS.dataset_dir, f"{dn}.h5py")
            dataset_loaded, metadata_loaded = ReplayBuffer.load_dataset_h5py(dataset_path)
            if dataset is not None:
                for key, value in dataset_loaded.items():
                    dataset[key] = np.concatenate((dataset[key], value), axis=0)
            else:
                dataset = dataset_loaded
        dataset = Dataset(dataset_dict=dataset)

    plot_episode_returns(
        dataset,
        bin=100,
        title=f"{FLAGS.dataset_name}",
        fig_path=f"{project_dir}/{FLAGS.dataset_name}.png")

    if FLAGS.filter_percentile is not None or FLAGS.filter_threshold is not None:
        dataset.filter(
            percentile=FLAGS.filter_percentile, threshold=FLAGS.filter_threshold
        )
    dataset.seed(FLAGS.seed)

    if "antmaze" in FLAGS.env_name:
        dataset.dataset_dict["rewards"] *= 100
    elif FLAGS.env_name.split("-")[0] in ["hopper", "halfcheetah", "walker2d"]:
        dataset.normalize_returns(scaling=1000)

    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop("cosine_decay", False):
        kwargs["decay_steps"] = FLAGS.max_steps
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(), **kwargs
    )

    # evaluate the initial agent
    eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
    eval_filepath = f"{project_dir}/eval_ave_episode_return.txt"
    with open(eval_filepath, "w") as f:
        expr_now = datetime.now()
        expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
        step = 0
        f.write(f"Experiment Time\tStep\tReturn\n")
        f.write(f"{expr_time_now_str}\t{step}\t{eval_info['return']}\n")
    eval_file = open(eval_filepath, "a")

    save_log(summary_writer, eval_info, 0, "evaluation", use_wandb=FLAGS.wandb)
    # save the initial agent
    initial_ckpt_filepath = f"{project_dir}/ckpts/ckpt_0"
    save_agent(agent, 0, initial_ckpt_filepath)
    # save the initial best agent
    if FLAGS.save_best:
        best_ckpt_filepath = f"{project_dir}/ckpts/best_ckpt"
        best_ckpt_performance = {"best_ckpt_return":eval_info["return"], "best_ckpt_step":0}
        save_agent(agent, 0, best_ckpt_filepath)

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        batch = dataset.sample(FLAGS.batch_size)
        info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            info = jax.device_get(info)
            save_log(summary_writer, info, i, "training", use_wandb=FLAGS.wandb)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=FLAGS.eval_episodes)
            # normalize the return: use the Reference MAX and MIN returns stored in
            # the registred environments defined in D4RL.
            # ToDO: use the MAX and MIN evaluation returns from online learning with the same seed
            if FLAGS.normalize_eval_return:
                eval_info["return"] = env.get_normalized_score(eval_info["return"]) * 100.0
            expr_now = datetime.now()
            expr_time_now_str = expr_now.strftime("%Y%m%d-%H%M%S")
            step = i
            eval_file.write(f"{expr_time_now_str}\t{step}\t{eval_info['return']}\n")
            eval_file.flush()
            save_log(summary_writer, eval_info, i, "evaluation", use_wandb=FLAGS.wandb)

            # save the current best agent
            if FLAGS.save_best and best_ckpt_performance["best_ckpt_return"] < eval_info["return"]:
                best_ckpt_performance["best_ckpt_return"] = eval_info["return"]
                best_ckpt_performance["best_ckpt_step"] = i
                save_agent(agent, i, best_ckpt_filepath)
            save_log(summary_writer, best_ckpt_performance, i, "evaluation", use_wandb=FLAGS.wandb)

    final_ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
    save_agent(agent, i, final_ckpt_filepath)

    if FLAGS.save_best:
        eval_file.write(f"**Best Policy**\t{best_ckpt_performance['best_ckpt_step']}\t{best_ckpt_performance['best_ckpt_return']}\n")
    eval_file.close()

    if not FLAGS.wandb: # close the tensorboard writer. wandb will close it automatically
        summary_writer.close()

if __name__ == "__main__":
    app.run(main)
