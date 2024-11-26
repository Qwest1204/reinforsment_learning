import gymnasium
import pygame
from vizdoom import gymnasium_wrapper
from gym.utils.play import play
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gymnasium.make("VizdoomCorridor-v0", render_mode="rgb_array"), keys_to_action=mapping)
# observation, info = env.reset()
# for _ in range(1000):  # this is where you would insert your policy
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

#     if terminated or truncated:
#         observation, info = env.reset()
#     env.render()
# env.close()