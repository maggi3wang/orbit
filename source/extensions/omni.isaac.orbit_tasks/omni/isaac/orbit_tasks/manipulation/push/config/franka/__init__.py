import gymnasium as gym
import os

from . import agents, joint_pos_env_cfg


gym.register(
    id="Isaac-Push-T-Block-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaTBlockPushEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-TBlock-Franka-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaTBlockPushEnvCfg_PLAY,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)