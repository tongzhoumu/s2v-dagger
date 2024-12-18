ALGO_NAME = 's2v-DAgger'
ENV_DOMAIN = 'Adroit'

import argparse
import os
import random
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict
from profiling import NonOverlappingTimeProfiler

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='test',
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="s2v",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="AdroitHandHammer-v1",
        help="the id of the environment")
    parser.add_argument("--expert-ckpt", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, default=20_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=500_000,
        help="the replay memory buffer size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps-per-collect", type=int, default=64,
        help="the number of steps to run in all environment in total per policy rollout")
    parser.add_argument("--minibatch-size", type=int, default=512,
        help="the size of mini-batches")
    parser.add_argument("--utd", type=float, default=0.5,
        help="Update-to-Data ratio (number of gradient updates / number of env steps)")
    parser.add_argument("--bc-loss-th", type=float, default=0.1, # important for training time
        help="if the bc loss is smaller than this threshold, then stop training and collect new data")
    parser.add_argument("--learning-starts", type=int, default=4000,
        help="timestep to start learning")

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--num-eval-envs", type=int, default=5)
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--log-freq", type=int, default=4000)
    parser.add_argument("--save-freq", type=int, default=1_000_000)

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.env_domain = ENV_DOMAIN
    args.script = __file__
    assert args.num_eval_envs == 1 or not args.capture_video, "Cannot capture video with multiple eval envs."
    assert args.num_steps_per_collect % args.num_envs == 0
    args.num_steps = int(args.num_steps_per_collect // args.num_envs)
    args.num_updates_per_collect = int(args.num_steps_per_collect * args.utd)
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    assert args.env_id in args.expert_ckpt, 'Expert checkpoint should be trained on the same env'
    assert (args.num_steps_per_collect * args.utd).is_integer()
    # fmt: on
    return args

from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class AdroitVisualStateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env: MujocoEnv
        w, h = self.env.unwrapped.width, self.env.unwrapped.height
        self.observation_space = gym.spaces.Dict(spaces={
            "rgb" : gym.spaces.Box(low=0, high=255, shape=(w, h, 3), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float32),
        })

    def observation(self, observation):
        return {
            "rgb": self.env.render(),
            "state": observation.astype(np.float32),
        }


def make_env(env_id, seed, video_dir=None, video_trigger=None, other_kwargs={}):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array", camera_name='fixed', width=128, height=128)
        # "fixed" camera is a closer view than "vil_camera" camera
        if video_dir:
            env = gym.wrappers.RecordVideo(env, video_dir, video_trigger)
        env = AdroitVisualStateWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 max_pooling=True,
                 inactivated_output=False, # False for ConvBody, True for CNN
                 ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),  # [128, 128]
            # nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=not inactivated_output)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=not inactivated_output)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.encoder = PlainConv(in_channels=3, out_dim=256, max_pooling=False, inactivated_output=False)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict
        self.forward = self.get_action = self.get_eval_action

    def get_feature(self, obs):
        # Preprocess the obs before passing to the real network, similar to the Dataset class in supervised learning
        img = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        return self.encoder(img)

    def get_eval_action(self, obs):
        x = self.get_feature(obs)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)

def unsqueeze(x, dim):
    if isinstance(x, dict):
        return {k: unsqueeze(v, dim) for k, v in x.items()}
    return x.unsqueeze(dim)

class DictArray(object):
    def __init__(self, buffer_shape, element_space, device, data_dict=None):
        self.buffer_shape = buffer_shape
        self.device = device
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device)
                else:
                    dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype).to(device)

    def keys(self):
        return self.data.keys()
    
    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
            return
        if isinstance(value, dict):
            for k, v in value.items():
                self.data[k][index] = v
        else:
            for k, v in value.data.items():
                self.data[k][index] = v
    
    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, device=self.device, data_dict=new_dict)
    
    # def to(self, device):
    #     return DictArray(self.buffer_shape, None, device=device, data_dict={
    #         k: v.to(device) for k, v in self.data.items()
    #     })


class DAggerBuffer(object):
    def __init__(self, buffer_size, collect_size, obs_space, action_space, device='cpu'):
        self.buffer_size = max(buffer_size // collect_size, 1)
        self.collect_size = collect_size
        self.observations = DictArray(buffer_shape=(self.buffer_size, collect_size), element_space=obs_space, device=device)
        self.expert_actions = torch.zeros((self.buffer_size, collect_size, int(np.prod(action_space.shape))), 
                                          dtype=torch.float32, device=device)
        self.device = device
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.buffer_size if self.full else self.pos

    def add(self, obs: DictArray, expert_actions):
        self.observations[self.pos] = obs
        self.expert_actions[self.pos] = expert_actions.clone().detach()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            batch_inds = (torch.randint(1, self.buffer_size, size=(batch_size,)) + self.pos) % self.buffer_size
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        # Sample randomly the env idx
        env_indices = torch.randint(0, high=self.collect_size, size=(len(batch_inds),))

        return dict(
            observations=self.observations[batch_inds, env_indices, :],
            expert_actions=self.expert_actions[batch_inds, env_indices, :],
        )


def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos: # infos is a dict
        indices = np.where(infos["_final_info"])[0] # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i] # info is also a dict
            ep = info['episode']
            print(f"global_step={global_step}, ep_return={ep['r'][0]:.2f}, ep_len={ep['l'][0]}, success={info['success']}")
            result['return'].append(ep['r'][0])
            result['len'].append(ep["l"][0])
            result['success'].append(info['success'])
    return result

def evaluate(n, agent, eval_envs, device):
    print('======= Evaluation Starts =========')
    agent.eval()
    result = defaultdict(list)
    obs, info = eval_envs.reset() # don't seed here
    while len(result['return']) < n:
        with torch.no_grad():
            action = agent.get_eval_action(to_tensor(obs, device))
        obs, rew, terminated, truncated, info = eval_envs.step(action.cpu().numpy())
        collect_episode_info(info, result)
    print('======= Evaluation Ends =========')
    agent.train()
    return result


import importlib.util
import sys

def import_file_as_module(path, module_name='tmp_module'):
    spec = importlib.util.spec_from_file_location(module_name, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
    return foo

if __name__ == "__main__":
    args = parse_args()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = '{:s}_{:d}'.format(now, args.seed)
    if args.exp_name: tag += '_' + args.exp_name
    log_name = os.path.join(ENV_DOMAIN, args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json
    with open(f'{log_path}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    os.environ['MUJOCO_GL'] = 'egl'
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv or args.num_envs == 1 \
        else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
    envs = VecEnv(
        [make_env(args.env_id, args.seed + i, other_kwargs=args.__dict__) for i in range(args.num_envs)]
    )
    VecEnv = gym.vector.SyncVectorEnv if args.sync_venv or args.num_eval_envs == 1 \
        else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
    eval_envs = VecEnv(
        [make_env(args.env_id, args.seed + 1000 + i,
                f'{log_path}/videos' if args.capture_video and i == 0 else None, 
                lambda x: x % (args.num_eval_episodes // args.num_eval_envs) == 0,
                other_kwargs=args.__dict__,
                ) 
        for i in range(args.num_eval_envs)]
    )
    eval_envs.reset(seed=args.seed+1000) # seed eval_envs here, and no more seeding during evaluation
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print(envs.single_action_space)
    print(envs.single_observation_space)

    # agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # expert setup
    from os.path import dirname as up
    expert_dir = up(up(args.expert_ckpt))
    import json
    with open(f'{expert_dir}/args.json', 'r') as f:
        expert_args = json.load(f)
    m = import_file_as_module(expert_args['script'])

    class DummyObject: pass
    dummy_env = DummyObject()
    dummy_env.single_observation_space = envs.single_observation_space['state']
    dummy_env.single_action_space = envs.single_action_space
    
    for key in ['Agent', 'Actor', 'QNetwork']:
        if hasattr(m, key):
            Expert = m.__dict__[key]
            break
    expert = Expert(dummy_env).to(device)
    checkpoint = torch.load(args.expert_ckpt)
    for key in ['agent', 'actor', 'q']:
        if key in checkpoint:
            expert.load_state_dict(checkpoint[key])
            break
    assert (expert.action_scale == agent.action_scale).all()
    assert (expert.action_bias == agent.action_bias).all()

    # DAgger buffer setup
    envs.single_observation_space.dtype = np.float32
    dagger_buf = DAggerBuffer(
        args.buffer_size,
        args.num_steps_per_collect,
        envs.single_observation_space,
        envs.single_action_space,
        device='cpu',
    )

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    next_obs = to_tensor(next_obs, device)
    num_updates = int(np.ceil(args.total_timesteps / args.num_steps_per_collect))
    result = defaultdict(list)
    timer = NonOverlappingTimeProfiler()

    for update in range(1, num_updates + 1):
        print('== Epoch:', update)

        agent.eval()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = agent.get_action(next_obs)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, _, _, _, info = envs.step(action.cpu().numpy())

            next_obs = to_tensor(next_obs, device)

            result = collect_episode_info(info, result)
        timer.end('collect')

        # flatten the batch
        b_obs = obs.reshape((-1,))
        # DAgger: save data to replay buffer
        b_expert_actions = expert.get_eval_action(b_obs['state']).detach()
        dagger_buf.add(b_obs, b_expert_actions)

        # Optimizing the policy and value network
        if global_step < args.learning_starts:
            continue

        agent.train()
        for i_update in range(args.num_updates_per_collect):
            # Behavior Cloning
            data = dagger_buf.sample(args.minibatch_size)
            pred_actions = agent.get_action(to_tensor(data['observations'], device))
            loss = F.mse_loss(pred_actions, data['expert_actions'].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _loss = loss.item()

            print('i_update:', i_update,'loss:', _loss)
            if args.bc_loss_th is not None and _loss < args.bc_loss_th:
                break

        timer.end('train')

        # Log
        if (global_step - args.num_steps_per_collect) // args.log_freq < global_step // args.log_freq:
            if len(result['return']) > 0:
                for k, v in result.items():
                    writer.add_scalar(f"train/{k}", np.mean(v), global_step)
                result = defaultdict(list)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/bc_loss", _loss, global_step)
            writer.add_scalar("losses/num_updates", i_update + 1, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            timer.dump_to_writer(writer, global_step)

        # Evaluation
        if (global_step - args.num_steps_per_collect) // args.eval_freq < global_step // args.eval_freq:
            result = evaluate(args.num_eval_episodes, agent, eval_envs, device)
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
            timer.end('eval')
        
        # Checkpoint
        if args.save_freq and ( update == num_updates or \
                (global_step - args.num_steps_per_collect) // args.save_freq < global_step // args.save_freq):
            os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
            torch.save({
                'agent': agent.state_dict(),
            }, f'{log_path}/checkpoints/{global_step}.pt')

    envs.close()
    writer.close()
