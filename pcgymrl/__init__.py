from gymnasium.envs.registration import register
from pcgymrl.envs.probs import PROBLEMS
from pcgymrl.envs.reps import REPRESENTATIONS
from pcgymrl.envs import PcgrlEnv
import gymnasium as gym

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    for rep in REPRESENTATIONS.keys():
        register(
            id="{}-{}-v0".format(prob, rep),
            entry_point="pcgymrl.envs:PcgrlEnv",
            kwargs={"prob": prob, "rep": rep},
        )


def make(game, render_mode="rgb_array") -> gym.Env:
    env = gym.make(game, render_mode=render_mode)
    return env
