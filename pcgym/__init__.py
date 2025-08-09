from gymnasium.envs.registration import register
from pcgym.envs.probs import PROBLEMS
from pcgym.envs.reps import REPRESENTATIONS
from pcgym.envs import PcgrlEnv
import gymnasium as gym

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    for rep in REPRESENTATIONS.keys():
        register(
            id="{}-{}-v0".format(prob, rep),
            entry_point="pcgym.envs:PcgrlEnv",
            kwargs={"prob": prob, "rep": rep},
        )


def make(game, render_mode="rgb_array") -> gym.Env:
    env = gym.make(game, render_mode=render_mode)
    return env
