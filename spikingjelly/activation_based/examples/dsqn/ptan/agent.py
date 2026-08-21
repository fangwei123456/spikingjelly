"""
Agent is something which converts states into actions and has state
"""

import copy
import numpy as np
import torch

from spikingjelly.activation_based import functional


class BaseAgent:
    """
    Abstract Agent interface
    """

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    return torch.tensor(np.asarray(states)).float() / 256


def _prepare_states(states, preprocessor, device):
    if preprocessor is None:
        return states
    states = preprocessor(states)
    return states.to(device) if torch.is_tensor(states) else states


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """

    def __init__(
        self,
        dqn_model,
        action_selector,
        device,
        preprocessor=default_states_preprocessor,
    ):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        states = _prepare_states(states, self.preprocessor, self.device)
        q_v = (
            self.dqn_model(states)
            if "dqn" in self.dqn_model.model_name
            else self.dqn_model.qvals(states)
        )
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        if "spiking" in self.dqn_model.model_name:
            functional.reset_net(self.dqn_model)
        return actions, agent_states


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model, target_model=None):
        self.model = model
        if target_model:
            self.target_model = target_model
        else:
            self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
