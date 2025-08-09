from pcgymrl.envs.probs import PROBLEMS
from pcgymrl.envs.reps import REPRESENTATIONS
from pcgymrl.envs.helper import get_int_prob, get_string_map
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import PIL

"""
The PCGRL GYM Environment
"""


class PcgrlEnv(gym.Env):
    """
    The type of supported rendering
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """

    def __init__(self, prob="binary", rep="narrow", render_mode=None):
        self.render_mode = render_mode
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(0.2 * self._prob._width * self._prob._height), 1)
        self._max_iterations = (
            self._max_changes * self._prob._width * self._prob._height
        )
        self._heatmap = np.zeros(
            (self._prob._height, self._prob._width), dtype=np.uint16
        )

        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space = self._rep.get_observation_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces["heatmap"] = spaces.Box(
            low=0,
            high=self._max_changes,
            dtype=np.uint16,
            shape=(self._prob._height, self._prob._width),
        )

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """

    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)
        return [seed]

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """

    def reset(self, seed=None, options=None):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(
            self._prob._width,
            self._prob._height,
            get_int_prob(self._prob._prob, self._prob.get_tile_types()),
        )
        self._rep_stats = self._prob.get_stats(
            get_string_map(self._rep._map, self._prob.get_tile_types())
        )
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros(
            (self._prob._height, self._prob._width), dtype=np.uint16
        )

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()

        return observation, {}  # Must return (obs, info)

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """

    def get_border_tile(self):
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """

    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """

    def adjust_param(self, **kwargs):
        if "change_percentage" in kwargs:
            percentage = min(1, max(0, kwargs.get("change_percentage")))
            self._max_changes = max(
                int(percentage * self._prob._width * self._prob._height), 1
            )
        self._max_iterations = (
            self._max_changes * self._prob._width * self._prob._height
        )
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space = self._rep.get_observation_space(
            self._prob._width, self._prob._height, self.get_num_tiles()
        )
        self.observation_space.spaces["heatmap"] = spaces.Box(
            low=0,
            high=self._max_changes,
            dtype=np.uint16,
            shape=(self._prob._height, self._prob._width),
        )

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """

    def step(self, action):
        self._iteration += 1
        # save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[y][x] = min(
                self._heatmap[y][x] + 1, self._max_changes
            )  # Clamp to max_changes
            self._rep_stats = self._prob.get_stats(
                get_string_map(self._rep._map, self._prob.get_tile_types())
            )
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)
        done = (
            self._prob.get_episode_over(self._rep_stats, old_stats)
            or self._changes >= self._max_changes
            or self._iteration >= self._max_iterations
        )
        terminated = self._prob.get_episode_over(self._rep_stats, old_stats)
        truncated = (
            self._changes >= self._max_changes
            or self._iteration >= self._max_iterations
        )
        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        # return the values
        return observation, reward, terminated, truncated, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """

    def render(self):
        img = self._prob.render(
            get_string_map(self._rep._map, self._prob.get_tile_types())
        )
        img = self._rep.render(
            img, self._prob._tile_size, self._prob._border_size
        ).convert("RGB")

        if self.render_mode == "rgb_array":
            return np.array(img)
        elif self.render_mode == "human":
            # For human mode, return None but you could add display logic here
            # For example, using matplotlib:
            # import matplotlib.pyplot as plt
            # plt.imshow(img)
            # plt.show(block=False)
            return None
        else:
            return None

    """
    Close the environment
    """

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
