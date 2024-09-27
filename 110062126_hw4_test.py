import gym
import numpy as np
import torch

class NoisyNetLayer(torch.nn.Module):
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
        self.mu_bias = torch.nn.Parameter(torch.zeros(out_features))
        self.mu_weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_bias = torch.nn.Parameter(torch.zeros(out_features))
        self.sigma_weight = torch.nn.Parameter(torch.zeros(out_features, in_features))

        self.cached_bias = None
        self.cached_weight = None

        self.register_noise_buffers()
        # self.parameter_initialization()
        # self.sample_noise()

    def forward(self, x, sample_noise=True):
        """
        Forward pass the layer. If training, sample noise depends on sample_noise.
        Otherwise, use the default weight and bias.
        """
        return torch.nn.functional.linear(x, weight=self.mu_weight, bias=self.mu_bias)

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

class LinearNet(torch.nn.Module):
    def __init__(self, layers, activation=torch.nn.ELU, layer_norm=False, linear_layer=torch.nn.Linear):
        # Create a simple Linear Network
        # layers[0] -> layers[1] -> ... -> layers[-2] -> layers[-1]
        super(LinearNet, self).__init__()
        self.input_shape = layers[0]
        self.output_shape = layers[-1]
        self.net = torch.nn.Sequential()

        for i in range(len(layers)-1):
            in_shape = layers[i]
            out_shape = layers[i+1]
            self.net.add_module(f'linear_{i}', linear_layer(in_shape, out_shape))
            if layer_norm == True:
                self.net.add_module(f'layer_norm_{i}', torch.nn.LayerNorm(out_shape))
            self.net.add_module(f'act_{i}', activation())

    def forward(self, x):
        x = self.net.forward(x)
        return x
        
class Actor(torch.nn.Module):
    def __init__(self, n_observation, n_action, layers,
                 activation=torch.nn.ELU, layer_norm=True, last_activation=torch.nn.Tanh):
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
    
    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            x = torch.from_numpy(observation)
        else:
            x = observation
        x = self.feature_net.forward(x)
        x = self.policy_net.forward(x)
        return x

def create_model():
    n_action = 22
    n_observation = 339
    actor_layers = "64-64"
    actor_layers = list(map(int, actor_layers.split('-')))
    actor_activation = torch.nn.ReLU
    actor_layer_norm = False

    actor = Actor(
        n_observation, n_action, actor_layers,
        activation=actor_activation,
        layer_norm=actor_layer_norm,
        last_activation=torch.nn.Tanh
    )
    return actor

def to_numpy(var):
    return var.cpu().data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.FloatTensor):
    return torch.autograd.Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype).to("cpu")

class Agent(object):
    def __init__(self):
        # Step1: Initialize actor and critic networks
        self.device = "cpu"
        self.actor = create_model()
        self.actor.to(self.device)
        self.load()
        self.frame_skip = 0
        self.last_action = None

    def act(self, observation):
        if self.frame_skip % 5 == 0:
            observation = self.modify_observation(observation)
            action = to_numpy(self.actor(to_tensor(np.array([observation], dtype=np.float32)))).squeeze(0)
            action = np.clip(action, -1.0, 1.0)
            action = action * 0.5 + 0.5
            self.last_action = action
        self.frame_skip += 1
        return self.last_action

    def modify_observation(self, observation):
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

    def load(self):
        data = torch.load(open("110062126_hw4_data", 'rb'), map_location=torch.device('cpu'))
        self.actor.load_state_dict(data)
        self.actor.eval()

if __name__ == '__main__':
    from osim.env import L2M2019Env
    from xml.etree import ElementTree as ET
    import importlib.util
    import sys
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    env = L2M2019Env(visualize=False,difficulty=2)

    # xml_file_path = 'meta.xml'
    # tree = ET.parse(xml_file_path)
    # root = tree.getroot()
    # sub_name = ""

    # for book in root.findall('info'):
    #     sub_name =  book.find('name').text

    # agent_path = sub_name + "_hw4_test.py"
    # module_name = agent_path.replace('/', '.').replace('.py', '')
    # spec = importlib.util.spec_from_file_location(module_name, agent_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    # spec.loader.exec_module(module)
    # Agent = getattr(module, 'Agent')

    import time
    from tqdm import tqdm

    total_reward = 0
    total_time = 0
    agent = Agent()
    time_limit = 120
    max_timesteps = env.spec.timestep_limit

    for episode in tqdm(range(10), desc="Evaluating"):
        obs = env.reset()
        start_time = time.time()
        episode_reward = 0
        timestep = 0
        
        while True:
            action = agent.act(obs) 
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            timestep += 1
            if timestep >= max_timesteps:
                print(f"Max timestep reached for episode {episode}")
                break
            if time.time() - start_time > time_limit:
                print(f"Time limit reached for episode {episode}")
                break
            if done:
                break
        end_time = time.time()
        total_reward += episode_reward
        total_time += (end_time - start_time)
    score = total_reward / 10
    print(f"Final Score: {score}")
