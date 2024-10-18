from VAE.load import prepare_image, init_model, device
import torch
import metaworld
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

def run_with_model():
    training_envs = init_metaworld45(render_mode="rgb_array", camera_id=1)
    done = False
    model = init_model("/home/qwest/project/PycharmProjects/Reinforsment_Learning/VAE.pth", 32, 1)
    while done != True:
        env = training_envs[4]
        obs = env.reset()  # Reset environment
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info, _ = env.step(a) 
        #print(env.action_space)
            # Display the resulting frame
        a = env.render()#[:, :, ::-1]
        a = torch.Tensor(prepare_image(a).unsqueeze(0))#.permute(1, 2, 0)
        reconstructed, mu, _ = model(a.to(device))
        reconstructed = reconstructed.view(-1, 3, 64, 64).detach().cpu().numpy().transpose(0, 2, 3, 1)
        print(reconstructed[0].shape)
        
        #fig = plt.figure(figsize=(25, 16))
        
        # for ii, img in enumerate(reconstructed):
        #     ax = fig.add_subplot(4, 8, ii + 1, xticks=[], yticks=[])
        plt.imshow(reconstructed[0])
        plt.show()
        
            # Press Q on keyboard to  exit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break 
        #done = True

def test_run():
    training_envs = init_metaworld45(render_mode="human", camera_id=2)
    done = False
    while done != True:
        env = training_envs[4]
        obs = env.reset()  # Reset environment
        a = env.action_space.sample()  # Sample an action
        obs, reward, done, info, _ = env.step(a) 
        env.render()
            # Press Q on keyboard to  exit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break 
        #done = True
#test_render_rgb_array()