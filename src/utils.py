"""Utility functions for the agent."""

import base64

import io
from pathlib import Path
import gymnasium as gym


import cv2


from IPython.display import display, HTML


def create_env(env_name: str) -> tuple[gym.Env, int, int]:
    """Create and return the environment."""
    env = gym.make(env_name, render_mode="rgb_array")
    state_shape = env.observation_space.shape
    state_size = state_shape[0]
    number_action = env.action_space.n

    print("State Shape :", state_shape)
    print("State Size :", state_size)
    print("Number of Actions :", number_action)

    return env, state_size, number_action


def show(file_path: str) -> None:
    """Display the video."""
    if Path(file_path).exists():
        # Display video
        mp4 = file_path
        video = io.open(mp4, "r+b").read()
        video = base64.b64encode(video)
        video = video.decode("ascii")
        video_tag = f"""<video muted autoplay width="640" height="480" controls>
                            <source src="data:video/mp4;base64,{video}" type="video/mp4">
                        </video>"""
        display(HTML(video_tag))

        video = cv2.VideoCapture(mp4)

        frame_per_second = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / frame_per_second if frame_per_second > 0 else 0

        print(f"frame per second={frame_per_second}")
        print(f"frame count={frame_count}")
        print(f"duration={duration}")
    else:
        print("Could not find video")
