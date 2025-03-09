from typing import NamedTuple, Optional, Tuple
import numpy as np
import gym
from gym import Env
from sim_transfer.hardware.car_env import CarEnv


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
    A gym env version of the SwitchCost Wrapper for wtc
    """

    def __init__(self,
                 env: CarEnv,
                 min_time_between_switches: float,
                 max_time_between_switches: float,
                 switch_cost: SwitchCost = ConstantSwitchCost(value=1.0),
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = True, ):
        """
           A Gym wrapper that integrates switch costs into the environment.
        """
        super().__init__()
        self.env = env
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        self.max_time_between_switches = max_time_between_switches
        self.sim_dt = 1.0 / 30
        assert time_as_part_of_state is True  # we keep this as true since we are going to wrap it with time for now
        assert min_time_between_switches >= self.sim_dt, \
            'Min steps between switches must be at least 1 / 30, this was the trained simulation dt'
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state  #this includes the state definition, for interaction cost time is part of the state
        self.state = None

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and return the initial augmented state
        """
        obs, info = self.env.reset(*args, **kwargs)  # Pass the args to CarEnv reset
        steps = np.array(0, dtype=int)  # Initialize number of passed steps as 0

        if self.time_as_part_of_state:  # we augment the state by the time component
            augmented_obs = np.concatenate([obs, steps.reshape(1)])
        else:  #we return the augmented pipeline state then
            augmented_obs = AugmentedPipelineState(pipeline_state=obs, steps=0)
        return augmented_obs, info

    @staticmethod
    def compute_steps(self,
                      pseudo_time: np.ndarray,
                      t_lower: float, #we now assume t_lower and t_upper describe discrete steps
                      t_upper: float) -> int:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2) #returns the time in seconds between [tmin, tmax]
        return np.floor(time_for_action // self.sim_dt)  # we scale back to number of steps

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        u, pseudo_time_for_action = action[:-1], action[-1]

        if self.state is None:
            self.state, _ = self.reset()
        # we calculate the number of applied steps, map the according action from [-1,1] to [step_min, step_max]
        steps_for_action = np.minimum(1, self.compute_steps(pseudo_time_for_action, self.min_time_between_switches,
                                                            self.max_time_between_switches))

        if self.time_as_part_of_state:
            obs, time = self.state[:-1], self.state[-1]  #time corresponds to the number of done steps * env.dt (so how much time has already passed)
        else:  #in this case, the state is an augmented pipeline state
            obs, time = self.state.pipeline_state, self.state.time

        # Fix (Arnav): we could also just use env.env_steps as a measure of how many steps have actually passed
        done = steps_for_action + self.env.env_steps >= self.env.max_steps  # retrieve how many steps have already passed

        num_steps = np.minimum(steps_for_action, self.env.max_steps - self.env.env_steps)

        total_reward = 0
        index = 0
        current_state = self.state

        # applying the number of steps in a loop
        while index < num_steps and not done:
            next_state, reward, done, info = self.env.step(
                action=u)  # each time we call step, the car system updates its env_steps, allowing for the car system to tell us if it has terminated
            total_reward += (self.discounting ** index) * (1 - done) * reward
            index += 1
            current_state = next_state

        total_reward = total_reward - self.switch_cost(state=self.state, action=u)

        # Augment state by time component
        new_time = (time + index * self.sim_dt) # updates the time we have had so far
        if self.time_as_part_of_state:
            augmented_next_state = np.concatenate([current_state, new_time])
        else:
            augmented_next_state = AugmentedPipelineState(pipeline_state=current_state, time=new_time)
        self.state = augmented_next_state  # update the state parameter
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

