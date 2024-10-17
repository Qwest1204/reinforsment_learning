import metaworld
import random
import cv2
import numpy as np
from VAE.load import prepare_image


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
    training_envs = init_metaworld45(render_mode="rgb_array", camera_id=1)
    done = False
    while done != True:
        env = training_envs[4]
        obs = env.reset()  # Reset environment
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info, _ = env.step(a) 
        print(env.action_space)
            # Display the resulting frame
        rgb_array = env.render()[:, :, ::-1]
        prepare_image(rgb_array)
        cv2.imshow('Frame',  rgb_array)
        
            # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break # Step the environment with the sampled random actio
        
        # [[[127 126 122] ----------- render rgb_array
        #   [127 126 122]
        #   [127 126 122]
        #   ...
        #   [127 126 122]
        #   [127 126 122]
        #   [127 126 122]]

        #  [[127 126 122]
        #   [127 126 122]
        #   [127 126 122]
        #   ...
        #   [127 126 122]
        #   [127 126 122]
        #   [127 126 122]]

        #  [[127 126 122]
        #   [127 126 122]
        #   [127 126 122]
        #   ...
        #   [127 126 122]
        #   [127 126 122]
        #   [127 126 122]]

        #  ...

        #  [[221 222 222]
        #   [217 217 217]
        #   [221 222 221]
        #   ...
        #   [218 219 218]
        #   [218 219 219]
        #   [221 221 221]]

        #  [[222 222 222]
        #   [222 222 222]
        #   [223 223 223]
        #   ...
        #   [218 219 219]
        #   [218 219 219]
        #   [219 219 219]]

        #  [[221 222 221]
        #   [221 222 221]
        #   [222 223 223]
        #   ...
        #   [219 219 219]
        #   [218 219 219]
        #   [217 218 218]]]


test_render_rgb_array()