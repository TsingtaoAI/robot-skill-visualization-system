# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrent(ActorCritic):
    is_recurrent: bool = True
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        activation: str = 'elu',
        rnn_type: str = 'lstm',
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        **kwargs: Any
    ) -> None:
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        if dones is None:
            return
        # Some envs provide dones/reset_buf as int/long 0/1; we need a boolean mask
        # for correct hidden-state reset semantics.
        if dones.dtype is not torch.bool:
            dones = dones.bool()
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(
        self,
        observations: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(
        self,
        critic_observations: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self) -> Tuple[Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[torch.Tensor, ...]]]:
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        type: str = 'lstm',
        num_layers: int = 1,
        hidden_size: int = 256
    ) -> None:
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn: Union[nn.LSTM, nn.GRU] = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    
    def forward(
        self,
        input: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones: Optional[torch.Tensor] = None) -> None:
        if dones is None or self.hidden_states is None:
            return
        # When the RNN is an LSTM, hidden_states is a tuple: (hidden_state, cell_state)
        # When the RNN is a GRU, it is a single tensor.
        states = self.hidden_states if isinstance(self.hidden_states, (tuple, list)) else (self.hidden_states,)
        for hidden_state in states:
            hidden_state[..., dones, :] = 0.0