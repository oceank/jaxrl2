#! /usr/bin/env python
import os
import gym
import tqdm
import orbax.checkpoint
from absl import app, flags
from ml_collections import config_flags
import matplotlib.pyplot as plt

from jaxrl2.agents import SACLearner
from jaxrl2.data import ReplayBuffer, plot_episode_returns
from jaxrl2.wrappers import wrap_gym
from jaxrl2.utils.save_load_agent import load_SAC_agent


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("run_name", "run", "The folder name for a saved run of policy learning.")
flags.DEFINE_string("replay_buffer_filename", "final_replay_buffer.h5py", "The filename of a saved replay buffer.")
flags.DEFINE_string("ckpt_name", "ckpt_0", "The name of the saved checkpoint where the policy is used as the behavior policy.")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_steps", int(1e6), "Number of new experiences to collect.")

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
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
    expr_run_dir = os.path.join(FLAGS.save_dir, FLAGS.run_name)

    # create the environment
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    # load the checkpoint
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    kwargs = dict(FLAGS.config)
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.num_steps
    )
    replay_buffer.seed(FLAGS.seed)

    ckpt_names = FLAGS.ckpt_name.split("_")
    new_experiences_per_agent = int(FLAGS.num_steps//len(ckpt_names))
    new_experiences_amount_list = [FLAGS.num_steps-new_experiences_per_agent*(len(ckpt_names)-1)] + [new_experiences_per_agent]*(len(ckpt_names)-1)
    print(f"Use the checkpoints: {ckpt_names}")
    print(f"New experiences to collect: {new_experiences_amount_list}")
    for ckpt_name, num_steps in zip(ckpt_names,new_experiences_amount_list):
        agent = SACLearner(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
        ckpt_filepath = f"{expr_run_dir}/ckpts/ckpt_{ckpt_name}"
        print(f"===> Loading the checkpoint {ckpt_filepath} to collect new {num_steps} experiences")
        load_SAC_agent(orbax_checkpointer, agent, ckpt_filepath)

        observation, done = env.reset(), False
        for i in tqdm.tqdm(
            range(1, num_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
        ):
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


    replay_buffer_metadata = {}
    replay_buffer_metadata["policy_learning_dir"] = FLAGS.run_name
    replay_buffer_metadata["policy_ckpt_name"] = FLAGS.ckpt_name
    replay_buffer_metadata["seed"] = FLAGS.seed
    replay_buffer_metadata["env_name"] = FLAGS.env_name
    experience_collection_tag = f"new_{FLAGS.num_steps}_experiences_by_{FLAGS.ckpt_name}"
    replay_buffer_filepath = f"{expr_run_dir}/{experience_collection_tag}.h5py"
    replay_buffer.save_dataset_h5py(replay_buffer_filepath, metadata=replay_buffer_metadata)
    plot_episode_returns(
        replay_buffer,
        bin=100,
        title=f"{experience_collection_tag}",
        fig_path=f"{expr_run_dir}/{experience_collection_tag}.png")

if __name__ == "__main__":
    app.run(main)
