from typing import NamedTuple, Optional, Tuple
import numpy as np
import gym
from gym import Env
class AugmentedPipelineState(NamedTuple):
    pipeline_state: np.ndarray
    time: float

class SwitchCost:
    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Computes the switch cost given state and value
        """
        raise NotImplementedError

class ConstantSwitchCost(SwitchCost):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.value

class IHSwitchCostWrapper(Env):
    """
    A gym env version of the SwithCost Wrapper for wtc
    """
    def __init__(self,
                 env: gym.Env,
                 num_integrator_steps: int,
                 min_time_between_switches: float,
                 max_time_between_switches: float | None = None,
                 switch_cost: SwitchCost = ConstantSwitchCost(value=1.0),
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = False,):
        """
           A Gym wrapper that integrates switch costs into the environment.
        """
        super().__init__()
        self.env = env
        self.num_integrator_steps = num_integrator_steps
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= self.env.dt, \
            'Min time between switches must be at least of the integration time dt' #otherwise the integration term makes no sense at all
        self.time_horizon = self.env.dt * self.num_integrator_steps #this corresponds to the T from the paper
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state #this includes the state definition, for interaction cost time is part of the state
        self.state = None
    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and return the initial augmented state
        """
        obs, info = self.env.reset(*args, **kwargs) # Pass the args to CarEnv reset
        time = np.array(0.0, dtype=np.float32) # Initialize time

        if self.time_as_part_of_state: # we augment the state by the time component
            augmented_obs = np.concatenate([obs, time.reshape(1)])
        else: #we return the augmented pipeline state then
            augmented_obs = AugmentedPipelineState(pipeline_state=obs, time=time)
        return augmented_obs, info

    def compute_time(self,
                     pseudo_time: np.ndarray,
                     dt: np.ndarray,
                     t_lower: np.ndarray,
                     t_upper: np.ndarray) -> np.ndarray:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time) + ((t_upper + t_lower) / 2)
        return np.floor(time_for_action / dt) * dt  # Ensure time is a multiple of dt

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        u, pseudo_time_for_action = action[:-1], action[-1]

        if self.state is None:
            self.state, _ = self.reset()
        # we calculate the application duration to the interval [tmin, tmax], action is [-1,1]
        time_for_action = self.compute_time(pseudo_time=pseudo_time_for_action,
                                            dt=self.dt,
                                            t_lower=self.min_time_between_switches,
                                            t_upper=self.max_time_between_switches
                                            )
        if self.time_as_part_of_state:
            obs, time = self.state[:-1], self.state[-1]
        else: #in this case, the state is an augmented pipeline state
            obs, time = self.state.pipeline_state, self.state.time

        done = time_for_action >= self.time_horizon - time

        num_steps = np.floor(np.minimum(time_for_action, self.time_horizon - time) / self.dt)

        current_state = self.state
        total_reward = 0
        index = 0

        #integration loop
        while index < num_steps and not done:
            next_state, reward, done, info = self.env.step(action=u)
            total_reward += reward
            current_state = next_state
            index += 1

        next_done = 1 - (1 - next_state.done) * (1 - done)

        total_reward = total_reward - self.switch_cost(state=self.state, action=u)

        # Prepare augmented obs
        next_time = (time + time_for_action).reshape(1)
        if self.time_as_part_of_state:
            augmented_next_state = np.concatenate([next_state, next_time])
        else:
            augmented_next_state = AugmentedPipelineState(pipeline_state=next_state, time = next_time)
        self.state = augmented_next_state #update the state parameter
        return augmented_next_state, total_reward, done, {}

    @property
    def state_dim(self) -> int:
        # +1 for time-to-go ant +1 for num remaining switches
        if self.time_as_part_of_state:
            return self.env.state_dim + 1
        else:
            return self.env.state_dim

    @property
    def action_dim(self) -> int:
        # +1 for time that we apply action for
        return self.env.action_dim + 1

    @property
    def dt(self):
        return self.env.dt


