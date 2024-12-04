import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.utils.play import play
num_eval_episodes = 4

env = gym.make("CarRacing-v3", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

play(env)

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')