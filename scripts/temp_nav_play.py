import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.navigator_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.nav_velocity_tracking import VelocityTrackingEasyEnv


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        # latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        # action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        action = body.forward(obs["obs_history"].to('cpu'))
        # info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        # print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        # print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # Cfg.env.num_actions = 3 # phase2

    from go1_gym.envs.wrappers.nav_history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy

    # policy = load_policy(logdir)

    return env, None


def play_go1(headless=True):
    from go1_gym import MINI_GYM_ROOT_DIR

    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "gait-conditioned-agility/2023-10-17/train" #phase2

    env, nav_policy = load_env(label, headless=headless)

    obs = env.reset()
    env.env.loco_obs_history = obs["loco_obs_history"]

    num_steps = 50
    actions = [[0.4,0.,0.], [0.,0.3,0.], [-0.4,0.,0.], [0.,-0.5,0.], [0.,0.,0.25]]

    for action in actions:
        action = torch.tensor(action).unsqueeze(0)
        for _ in range(num_steps):
            obs, rew, done, info = env.step(action)
            # print(rew.shape,rew)
            # print(done.shape,done)
            env.env.loco_obs_history = obs["loco_obs_history"]



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
