import torch
from legged_gym.simulator import GenesisSimulator, IsaacGymSimulator, IsaacLabSimulator
from legged_gym import SIMULATOR
from typing import Any, Dict, Optional, Tuple, Protocol

# Lightweight protocol describing the minimal config interface used by BaseTask
class _EnvCfg(Protocol):
    num_envs: int
    num_observations: int
    num_privileged_obs: Optional[int]
    num_actions: int

class _Cfg(Protocol):
    env: _EnvCfg

EnvCfg = _EnvCfg  # alias for clearer type hints in annotations
Cfg = _Cfg

# Type alias for simulator instance type (used for a class attribute annotation)
SimulatorInstance = "Simulator"  # Forward reference to Simulator ABC

# Base class for RL tasks
class BaseTask:

    # Public buffers and basic state (annotated for static typing)
    obs_buf: torch.Tensor
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor
    time_out_buf: torch.Tensor
    privileged_obs_buf: Optional[torch.Tensor]
    extras: Dict[str, Any]

    # Environment/algorithm interface metadata
    num_envs: int
    num_actions: int
    num_obs: int
    num_privileged_obs: Optional[int]
    simulator: SimulatorInstance
    device: torch.device
    headless: bool

    def __init__(self, cfg: Cfg, sim_params: Any, sim_device: torch.device, headless: bool) -> None:
        
        self.render_fps = 50
        self.last_frame_time = 0

        self.device = sim_device
        self.headless = headless

        # read basic config (typing hints rely on cfg.env having these attributes)
        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}
        
        if SIMULATOR == "genesis":
            self.simulator = GenesisSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaacgym":
            self.simulator = IsaacGymSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaaclab":
            self.simulator = IsaacLabSimulator(cfg, sim_params, sim_device, self.headless)
        else:
            raise ValueError(f"Unknown simulator: {SIMULATOR}")

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf
    
    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return self.privileged_obs_buf
    
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError
