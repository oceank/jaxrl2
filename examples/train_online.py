#! /usr/bin/env python
import os
import gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_load_agent import save_SAC_agent, load_SAC_agent, equal_SAC_agents
from jaxrl2.utils.save_expr_log import save_log

from tensorboardX import SummaryWriter
from datetime import datetime

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("ckpt_interval", 10000, "Checkpoint interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
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

# Path 1: purely online (on)
# Path 2: online2offline (on2of)
# path 3: online2offline2online (on2of2on)
def main(_):
    now = datetime.now()
    expr_time_str = now.strftime("%Y%m%d-%H%M%S")
    project_name = f"{FLAGS.env_name}_seed{FLAGS.seed}_on_sac_{expr_time_str}"
    project_dir = os.path.join(FLAGS.save_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)
    if FLAGS.wandb:
        summary_writer = wandb.init(project=project_name)
        summary_writer.config.update(FLAGS)
    else:
        summary_writer = SummaryWriter(project_dir, write_to_disk=True)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)

    # evaluate the initial agent
    eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
    save_log(summary_writer, eval_info, 0, "evaluation", use_wandb=FLAGS.wandb)
    # save the initial best agent
    if FLAGS.save_best:
        best_ckpt_filepath = f"{project_dir}/ckpts/best_ckpt"
        best_ckpt_performance = {"best_ckpt_return":eval_info["return"], "best_ckpt_step":0}
        save_SAC_agent(agent, 0, best_ckpt_filepath)
    # save the checkpoint at step 0: the initial agent
    if FLAGS.save_ckpt:
        ckpt_filepath = f"{project_dir}/ckpts/ckpt_0"
        save_SAC_agent(agent, 0, ckpt_filepath)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
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
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                save_log(summary_writer, update_info, i, "training", use_wandb=FLAGS.wandb)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            save_log(summary_writer, eval_info, i, "evaluation", use_wandb=FLAGS.wandb)

            # save the initial best agent
            if FLAGS.save_best and best_ckpt_performance["best_ckpt_return"] < eval_info["return"]:
                best_ckpt_performance["best_ckpt_return"] = eval_info["return"]
                best_ckpt_performance["best_ckpt_step"] = i
                save_SAC_agent(agent, i, best_ckpt_filepath)
            # save the checkpoint at step i
            if FLAGS.save_ckpt and (i % FLAGS.ckpt_interval == 0):
                ckpt_filepath = f"{project_dir}/ckpts/ckpt_{i}"
                save_SAC_agent(agent, i, ckpt_filepath)

                # saneity check for saving and loading
                #agent2 = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
                #load_SAC_agent(agent2, ckpt_filepath)
                #print(f"Are the agents equal? {equal_SAC_agents(agent, agent2)}")
            save_log(summary_writer, best_ckpt_performance, i, "evaluation", use_wandb=FLAGS.wandb)

if __name__ == "__main__":
    app.run(main)
