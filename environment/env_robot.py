import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import metaworld
import random
import cv2
import numpy as np
import VAE.utils as u
from gym import spaces
from stable_baselines3 import DQN


model = u.init_model(path_to_weight='/home/qwest/project/PycharmProjects/Reinforsment_Learning/VAE/ROBOT.pt', latent_dim=32, batch_size=1)


def init_metaworld10(render_mode, camera_id):
    ml10 = metaworld.ML10() # Construct the benchmark, sampling tasks

    training_envs = []
    for name, env_cls in ml10.train_classes.items():
        env = env_cls(render_mode=render_mode, camera_id=camera_id)
        task = random.choice([task for task in ml10.train_tasks
                                if task.env_name == name])
        env.set_task(task)
        training_envs.append(env)
    return training_envs

def init_metaworld45(render_mode, camera_id):
    ml45 = metaworld.ML45() # Construct the benchmark, sampling tasks

    training_envs = []
    for name, env_cls in ml45.train_classes.items():
        env = env_cls(render_mode=render_mode, camera_id=camera_id)
        task = random.choice([task for task in ml45.train_tasks
                                if task.env_name == name])
        env.set_task(task)
        training_envs.append(env)
    return training_envs

def test_render_rgb_array():
    training_envs = init_metaworld45(render_mode="rgb_array", camera_id=2)
    done = False
    env = training_envs[4]
    action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float16)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_cartpole")
    while done != True:
        obs = env.reset()
        #action, _states = model.predict(obs, deterministic=True)
         # Reset environment
        action = env.action_space.sample()   # Sample an action
        print(action)
        obs, reward, done, info, _ = env.step(action) 
        #print(env.action_space)
            # Display the resulting frame 
        frame = env.render()[:, :, ::-1] ##  <class 'numpy.ndarray'>  (480, 480, 3)
        ij = u.compute_image(model=model, img=frame)
        cv2.imshow('Frame',  ij[0])
        
            # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break # Step the environment with the sampled random actio
test_render_rgb_array()