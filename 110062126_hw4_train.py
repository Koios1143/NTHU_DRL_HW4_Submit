from osim.env import L2M2019Env
import numpy as np
import gym
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
import datetime
from collections import deque
from tqdm import trange
import random
import wandb
from torch.autograd import Variable

device = 'cuda'

def boolean_flag(parser, name, default=False, help=None):
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--skip-frames', type=int, default=5)
    parser.add_argument('--fail-reward', type=float, default=-0.2)
    parser.add_argument('--reward-scale', type=float, default=10.)
    
    for agent in ["actor", "critic"]:
        parser.add_argument('--{}-layers'.format(agent), type=str, default="64-64")
        parser.add_argument('--{}-activation'.format(agent), type=str, default="relu")
        boolean_flag(parser, "{}-layer-norm".format(agent), default=False)
        
        parser.add_argument('--{}-lr'.format(agent), type=float, default=1e-3)
        parser.add_argument('--{}-lr-end'.format(agent), type=float, default=5e-5)
    
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--loss-type', type=str, default="quadric-linear")
    parser.add_argument('--grad-clip', type=float, default=10.)

    parser.add_argument('--tau', default=0.0001, type=float)

    parser.add_argument('--train-steps', type=int, default=int(1e4))
    parser.add_argument('--batch-size', type=int, default=256)

    parser.add_argument('--buffer-size', type=int, default=int(1e6))

    parser.add_argument('--initial-epsilon', default=0.5, type=float)
    parser.add_argument('--final-epsilon', default=0.001, type=float)
    parser.add_argument('--max-episodes', default=int(1e4), type=int)
    parser.add_argument('--max-update-steps', default=int(5e6), type=int)
    parser.add_argument('--epsilon-cycle-len', default=int(2e2), type=int)

    parser.add_argument('--rp-type', default="ornstein-uhlenbeck", type=str)
    parser.add_argument('--rp-theta', default=0.15, type=float)
    parser.add_argument('--rp-sigma', default=0.2, type=float)
    parser.add_argument('--rp-sigma-min', default=0.15, type=float)
    parser.add_argument('--rp-mu', default=0.0, type=float)

    parser.add_argument('--clip-delta', default=10, type=int)
    parser.add_argument('--save-step', default=int(1e4), type=int)

    return parser.parse_args()

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, args):
        super(EnvWrapper, self).__init__(env)
        self.skip_frames = args.skip_frames
        self.reward_scale = args.reward_scale
        self.fail_reward = args.fail_reward
        
        # The output of network in range [-1, 1] (Tanh func.)
        # We want it to be [0, 1]
        action_mean, action_std = 0.5, 0.5
        self.normalize_action = lambda x: (x - action_mean) / action_std
        self.denormalize_action = lambda x: x * action_std + action_mean
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self.observation(observation)
        self.env_step = 0
        return observation

    def step(self, action):
        # [-1, 1] -> [0, 1]
        action = self.denormalize_action(action)
        
        # Skip Frames
        total_reward = 0
        for _ in range(self.skip_frames):
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.env_step += 1
            if done:
                if self.env_step < 1000:
                    total_reward += self.fail_reward
                break
        
        total_reward *= self.reward_scale
        obs = self.observation(obs)
        return obs, total_reward, done, None
    
    def observation(self, observation):
        # Flatten dict into one numpy array with shape (339, )
        # v_tgt_field
        a = observation['v_tgt_field'].flatten()
        # pelvis
        b = [val for val in observation['pelvis'].values() if isinstance(val, float)]
        b.extend(observation['pelvis']['vel'])
        b = np.array(b)
        # r_leg
        c = np.array([val for v in observation['r_leg'].values() for val in (v if isinstance(v, list) else v.values())])
        # l_leg
        d = np.array([val for v in observation['l_leg'].values() for val in (v if isinstance(v, list) else v.values())])
        return np.concatenate((a, b, c, d)).astype('float32')

def create_env(args, visualize=False):
    env = L2M2019Env(visualize=visualize, difficulty=2, seed=args.seed)
    env = EnvWrapper(env, args)
    return env

import torch
import torch.nn as nn

class NoisyNetLayer(nn.Module):
    """
    NoisyNet layer, factorized version
    """
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()

        self.sigma = sigma
        self.in_features = in_features
        self.out_features = out_features

        # mu_b, mu_w, sigma_b, sigma_w
        # size: q, qxp, q, qxp
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))

        self.cached_bias = None
        self.cached_weight = None

        self.register_noise_buffers()
        self.parameter_initialization()
        self.sample_noise()

    def forward(self, x, sample_noise=True):
        """
        Forward pass the layer. If training, sample noise depends on sample_noise.
        Otherwise, use the default weight and bias.
        """
        if self.training:
            if sample_noise:
                self.sample_noise()
            return nn.functional.linear(x, weight=self.weight, bias=self.bias)
        else:
            return nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

    def register_noise_buffers(self):
        """
        Register noise f(epsilon_in) and f(epsilon_out)
        """
        self.register_buffer(name='epsilon_input', tensor=torch.empty(self.in_features))
        self.register_buffer(name='epsilon_output', tensor=torch.empty(self.out_features))

    def _calculate_bound(self):
        """
        Determines the initialization bound for the FactorisedNoisyLayer based on the inverse
        square root of the number of input features. This approach to determining the bound
        takes advantage of the factorised noise model's efficiency and aims to balance the
        variance of the outputs relative to the variance of the inputs. Ensuring that the
        initialization of weights does not saturate the neurons and allows for stable
        gradients during the initial phases of training.
        """
        return self.in_features**(-0.5)

    @property
    def weight(self):
        """
        w = sigma \circ epsilon + mu
        epsilon = f(epsilon_in)f(epsilon_out)
        """
        if self.cached_weight is None:
            self.cached_weight = self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight
        return self.cached_weight

    @property
    def bias(self):
        """
        b = sigma \circ epsilon + mu
        """
        if self.cached_bias is None:
            self.cached_bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return self.cached_bias

    def sample_noise(self):
        """
        Sample factorised noise
        f(x) = sgn(x)\sqrt{|x|}
        """
        with torch.no_grad():
            epsilon_input = torch.randn(self.in_features, device=self.epsilon_input.device)
            epsilon_output = torch.randn(self.out_features, device=self.epsilon_output.device)
            self.epsilon_input = (epsilon_input.sign() * torch.sqrt(torch.abs(epsilon_input))).clone()
            self.epsilon_output = (epsilon_output.sign() * torch.sqrt(torch.abs(epsilon_output))).clone()
        self.cached_weight = None
        self.cached_bias = None

    def parameter_initialization(self):
        """
        Initialize with normal distribution
        """
        bound = self._calculate_bound()
        self.sigma_bias.data.fill_(value=self.sigma * bound)
        self.sigma_weight.data.fill_(value=self.sigma * bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.mu_weight.data.uniform_(-bound, bound)

class LinearNet(nn.Module):
    def __init__(self, layers, activation=torch.nn.ELU, layer_norm=False, linear_layer=nn.Linear):
        # Create a simple Linear Network
        # layers[0] -> layers[1] -> ... -> layers[-2] -> layers[-1]
        super(LinearNet, self).__init__()
        self.input_shape = layers[0]
        self.output_shape = layers[-1]
        self.net = nn.Sequential()

        for i in range(len(layers)-1):
            in_shape = layers[i]
            out_shape = layers[i+1]
            self.net.add_module(f'linear_{i}', linear_layer(in_shape, out_shape))
            if layer_norm == True:
                self.net.add_module(f'layer_norm_{i}', nn.LayerNorm(out_shape))
            self.net.add_module(f'act_{i}', activation())

    def forward(self, x):
        x = self.net.forward(x)
        return x
        
class Actor(nn.Module):
    def __init__(self, n_observation, n_action, layers,
                 activation=nn.ELU, layer_norm=True, last_activation=nn.Tanh, init_w=3e-3):
        super(Actor, self).__init__()

        linear_layer = NoisyNetLayer
        self.feature_net = LinearNet(
            layers=[n_observation] + layers,
            activation=activation,
            layer_norm=layer_norm,
            linear_layer=linear_layer)
        self.policy_net = LinearNet(
            layers=[self.feature_net.output_shape, n_action],
            activation=last_activation,
            layer_norm=False
        )
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')

        for layer in self.policy_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            x = torch.from_numpy(observation)
        else:
            x = observation
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)
        return x

class Critic(nn.Module):
    def __init__(self, n_observation, n_action, layers,
                 activation=nn.ELU, layer_norm=False, init_w=3e-3):
        super(Critic, self).__init__()
        linear_layer = NoisyNetLayer
        self.feature_net = LinearNet(
            layers=[n_observation + n_action] + layers,
            activation=activation,
            layer_norm=layer_norm,
            linear_layer=linear_layer)
        self.value_net = nn.Linear(self.feature_net.output_shape, 1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        for layer in self.feature_net.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')
        self.value_net.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, observation, action):
        observation = torch.from_numpy(observation) if isinstance(observation, np.ndarray) else observation
        action = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        x = torch.cat((observation, action), dim=1)
        x = self.feature_net.forward(x)
        x = self.value_net.forward(x)
        return x

class ReplayBuffer(object):
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.max_size = size
    
    def __len__(self):
        return len(self.buffer)

    def store(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def sample(self, batch_size):
        indice = np.random.randint(0, len(self.buffer), size=batch_size)
        
        obses, actions, rewards, next_obses, dones = [], [], [], [], []
        for i in indice:
            data = self.buffer[i]
            obs, action, reward, next_obs, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_obses.append(np.array(next_obs, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(next_obses), np.array(dones)

class AnnealedGaussianProcess(object):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing=int(1e5)):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    def reset_states(self):
        pass

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2,
                 x0=None, size=1, sigma_min=None, n_steps_annealing=int(1e5)):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

def create_decay_fn(decay_type, initial_value, final_value, max_step=None, cycle_len=None, num_cycles=None):
    if decay_type == "linear":
        def decay_fn(step):
            relative = 1. - step / max_step
            return initial_value * relative + final_value * (1. - relative)
        return decay_fn
    else:
        max_step = cycle_len * num_cycles
        def decay_fn(step):
            relative = 1. - step / max_step
            relative_cosine = 0.5 * (np.cos(np.pi * np.mod(step, cycle_len) / cycle_len) + 1.0)
            return relative_cosine * (initial_value - final_value) * relative + final_value
        return decay_fn

def to_numpy(var):
    return var.cpu().data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.cuda.FloatTensor):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype).to(device)

activations = {
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leakyrelu": torch.nn.LeakyReLU,
    "selu": torch.nn.SELU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh
}

def create_model(args):
    actor = Actor(
        args.n_observation, args.n_action, args.actor_layers,
        activation=args.actor_activation,
        layer_norm=args.actor_layer_norm,
        last_activation=nn.Tanh
    )
    critic = Critic(
        args.n_observation, args.n_action, args.critic_layers,
        activation=args.critic_activation,
        layer_norm=args.critic_layer_norm
    )
    return actor, critic

class QuadricLinearLoss(nn.Module):
    def __init__(self, clip_delta):
        super(QuadricLinearLoss, self).__init__()
        self.clip_delta = clip_delta

    def forward(self, y_pred, y_true, weights):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        loss = torch.mean(loss * weights)
        return loss

class Agent(object):
    def __init__(self, args, env, eval_env):
        self.save_dir = Path(args.logdir) / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.save_dir.mkdir(parents=True)
        self.env = env
        self.eval_env = eval_env

        with open(self.save_dir / "args.json", "w") as fout:
            json.dump(vars(args), fout, indent=4, ensure_ascii=False, sort_keys=True)
        
        self.run = wandb.init(
            entity="koioslin",
            project="DRL_HW4",
            config=vars(args)
        )

        args.n_action = env.action_space.shape[0]
        args.n_observation = env.observation_space.shape[0]

        args.actor_layers = list(map(int, args.actor_layers.split('-')))
        args.critic_layers = list(map(int, args.critic_layers.split('-')))

        args.actor_activation = activations[args.actor_activation]
        args.critic_activation = activations[args.critic_activation]

        self.args = args

        # Step1: Initialize actor and critic networks
        self.actor, self.critic = create_model(args)
        self.actor.train()
        self.critic.train()
        self.actor.to(device)
        self.critic.to(device)
        self.actor_lr_decay_fn = create_decay_fn("linear", initial_value=args.actor_lr, final_value=args.actor_lr_end, max_step=args.max_update_steps)
        self.critic_lr_decay_fn = create_decay_fn("linear", initial_value=args.critic_lr, final_value=args.critic_lr_end, max_step=args.max_update_steps)

        # Step2: Initialize target actor and target critic networks
        self.target_actor, self.target_critic = create_model(args)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.to(device)
        self.target_critic.to(device)

        # Step3: Initialize Replay Buffer
        self.buffer = ReplayBuffer(args.buffer_size)

        self.random_process = OrnsteinUhlenbeckProcess(size=args.n_action, theta=args.rp_theta, mu=args.rp_mu, sigma=args.rp_sigma, sigma_min=args.rp_sigma_min)
        self.total_steps = 0
        self.update_steps = 0
        self.epsilon_cycle_len = random.randint(args.epsilon_cycle_len // 2, args.epsilon_cycle_len * 2)
        self.epsilon_decay_fn = create_decay_fn(
            "cycle",
            initial_value=args.initial_epsilon,
            final_value=args.final_epsilon,
            cycle_len=self.epsilon_cycle_len,
            num_cycles=args.max_episodes // self.epsilon_cycle_len)
    
    def train(self):
        args = self.args
        env = self.env
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        ep_rewards = []
        # Loop episodes
        for self.episode in trange(args.max_episodes):
            # Step4: Initialize a random process N, here we use Ornstein-Uhlenbeck Process
            self.random_process.reset_states()
            # Step5: Receive initial observation state
            seed = random.randrange(2 ** 32 - 2)
            observation = env.reset(seed=seed)
            done = False
            
            self.actor_lr = self.actor_lr_decay_fn(self.update_steps)
            self.critic_lr = self.critic_lr_decay_fn(self.update_steps)
            self.actor_lr = min(args.actor_lr, max(args.actor_lr_end, self.actor_lr))
            self.critic_lr = min(args.critic_lr, max(args.critic_lr_end, self.critic_lr))
            
            self.criterion = QuadricLinearLoss(clip_delta=args.clip_delta)

            self.epsilon = min(args.initial_epsilon, max(args.final_epsilon, self.epsilon_decay_fn(self.episode)))
            ep_reward = 0
            ep_critic_losses = []
            ep_policy_losses = []
            ep_steps = 0

            while not done:
                # Step6: Select action according to current policy
                action = self.act(observation, noise=self.epsilon * self.random_process.sample())
                next_observation, reward, done, _ = env.step(action)
                ep_reward += reward
                # Step8: Store transition in Replay Buffer
                self.buffer.store(observation, action, reward, next_observation, done)
                # Step9: Sample a random minibatch of N transitions from Replay Buffer
                if self.total_steps >= args.train_steps:
                    observations, actions, rewards, next_observations, dones = self.buffer.sample(batch_size=args.batch_size)
                    # Step10: Update
                    metrics, info = self.update(observations, actions, rewards, next_observations, dones, self.actor_lr, self.critic_lr)
                    ep_critic_losses.append(to_numpy(metrics['value_loss']))
                    ep_policy_losses.append(to_numpy(metrics['policy_loss']))
                    self.update_steps += 1

                    if self.update_steps % args.save_step == 0:
                        with torch.no_grad():
                            self.save(self.update_steps)
                            self.evaluate()
                
                self.run.log({
                    "episode": self.episode,
                    "total_steps": self.total_steps,
                    "update_steps": self.update_steps,
                    "epsilon": self.epsilon,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr,
                    "buffer_size": len(self.buffer)
                })
                
                self.total_steps += 1
                ep_steps += 1
                observation = next_observation
            ep_rewards.append(ep_reward)
            self.run.log({
                "episode": self.episode,
                "total_steps": self.total_steps,
                "update_steps": self.update_steps,
                "mean_ep_reward": np.mean(ep_rewards),
                "ep_reward": ep_reward,
                "mean_ep_critic_loss": np.mean(ep_critic_losses),
                "mean_ep_policy_loss": np.mean(ep_policy_losses),
                "ep_steps": ep_steps
            })
    
    def act(self, observation, noise=0):
        action = to_numpy(self.actor(to_tensor(np.array([observation], dtype=np.float32)))).squeeze(0)
        action += noise
        action = np.clip(action, -1.0, 1.0)
        return action

    def update(self,
            observations, actions, rewards, next_observations, dones,
            actor_lr=1e-4, critic_lr=1e-3):
        
        args = self.args
        dones = dones[:, None].astype(np.bool)
        rewards = rewards[:, None].astype(np.float32)
        weights = np.ones_like(rewards)

        dones = to_tensor(np.invert(dones).astype(np.float32))
        rewards = to_tensor(rewards)
        weights = to_tensor(weights, requires_grad=False)

        # Step10: Set y_i = r_i + Q'(s_{i+1}, \mu'(s_{i+1}|\theta^{\mu'})|\theta^{Q'})
        next_v_values = self.target_critic(
            to_tensor(next_observations, volatile=True),
            self.target_actor(to_tensor(next_observations, volatile=True)),
        )
        with torch.no_grad():
            reward_predicted = dones * args.gamma * next_v_values
        td_target = rewards + reward_predicted

        # Step11: Update critic
        self.critic.zero_grad()
        v_values = self.critic(to_tensor(observations), to_tensor(actions))
        value_loss = self.criterion(v_values, td_target, weights=weights)
        value_loss.backward()

        torch.nn.utils.clip_grad_norm(self.critic.parameters(), args.grad_clip)
        for param_group in self.critic_optim.param_groups:
            param_group["lr"] = critic_lr

        self.critic_optim.step()

        # Step12: Update policy
        self.actor.zero_grad()
        policy_loss = -self.critic(
            to_tensor(observations),
            self.actor(to_tensor(observations))
        )
        policy_loss = torch.mean(policy_loss * weights)
        policy_loss.backward()

        torch.nn.utils.clip_grad_norm(self.actor.parameters(), args.grad_clip)
        for param_group in self.actor_optim.param_groups:
            param_group["lr"] = actor_lr

        self.actor_optim.step()

        # Step13: Update target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - args.tau) + param.data * args.tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - args.tau) + param.data * args.tau)
        

        metrics = {
            "value_loss": value_loss,
            "policy_loss": policy_loss
        }

        td_v_values = self.critic(
            to_tensor(observations, volatile=True, requires_grad=False),
            to_tensor(actions, volatile=True, requires_grad=False))
        td_error = td_target - td_v_values

        info = {
            "td_error": to_numpy(td_error)
        }

        self.run.log({
            "episode": self.episode,
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
            "critic_loss": value_loss,
            "policy_loss": policy_loss,
            "mean_td_error": np.mean(to_numpy(td_error))
        })

        return metrics, info

    def evaluate(self):
        eval_env = self.eval_env
        self.actor.eval()
        observation = eval_env.reset()
        done = False
        ep_reward = 0
        ep_critic_losses = []
        ep_policy_losses = []
        ep_steps = 0

        while not done:
            action = self.act(observation)
            next_observation, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            ep_steps += 1
            observation = next_observation
    
        self.run.log({
            "episode": self.episode,
            "total_steps": self.total_steps,
            "update_steps": self.update_steps,
            "eval_ep_reward": ep_reward,
            "eval_mean_ep_critic_loss": np.mean(ep_critic_losses),
            "eval_mean_ep_policy_loss": np.mean(ep_policy_losses),
            "eval_ep_steps": ep_steps
        })
        
        self.actor.train()
    
    def save(self, episode=None):
        save_path = self.save_dir / f"episode_{episode}"
        save_path.mkdir(parents=True)
        torch.save(self.actor.state_dict(), "{}/actor_state_dict.pkl".format(save_path))
        torch.save(self.critic.state_dict(), "{}/critic_state_dict.pkl".format(save_path))
        torch.save(self.target_actor.state_dict(), "{}/target_actor_state_dict.pkl".format(save_path))
        torch.save(self.target_critic.state_dict(), "{}/target_critic_state_dict.pkl".format(save_path))

    def load(self, path="./logs/2024-05-11T00-54-33/episode_204000"):
        self.actor.load_state_dict(torch.load(open(f"{path}/actor_state_dict.pkl", 'rb')))
        self.critic.load_state_dict(torch.load(open(f"{path}/critic_state_dict.pkl", 'rb')))
        self.target_actor.load_state_dict(torch.load(open(f"{path}/target_actor_state_dict.pkl", 'rb')))
        self.target_critic.load_state_dict(torch.load(open(f"{path}/target_critic_state_dict.pkl", 'rb')))
        
if __name__ == '__main__':
    args = parse_args()
    env = create_env(args)
    eval_env = create_env(args, visualize=True)
    agent = Agent(args=args, env=env, eval_env=eval_env)
    agent.load()
    agent.train()