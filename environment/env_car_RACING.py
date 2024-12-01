import gym
from gym.envs.box2d import CarRacing

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

if __name__=='__main__':
    env = lambda :  CarRacing(
        #grayscale=1,
        #show_info_panel=0,
        #discretize_actions="hard",
        #frames_per_state=4,
        #num_lanes=1,
        #num_tracks=1,
        )

    #env = getattr(environments, env)
    env = DummyVecEnv([env])

    model = PPO.load('/home/qwest/project/PycharmProjects/Reinforsment_Learning/environment/car_racing_weights.pkl.zip')

    model.set_env(env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()