import torch
import math
import numpy as np
import torch.nn as nn
from collections import defaultdict
from lib.rl.core.running_norm import RunningNorm
from lib.models.mlp import MLP
from lib.rl.core.distributions import Categorical, DiagGaussian

from custom.models.sgnn import SGNN

class DevPolicy(nn.Module):
    def __init__(self, cfg, agent):
        super(DevPolicy, self).__init__()
        self.cfg = cfg
        self.agent = agent
        # dimension define
        self.scale_state_dim = agent.scale_state_dim
        self.sim_action_dim = agent.sim_action_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.action_dim = agent.action_dim
        self.state_dim = agent.state_dim

        # scale transform
        self.scale_norm = RunningNorm(self.scale_state_dim)
        cur_dim = self.scale_state_dim
        self.scale_mlp = MLP(cur_dim,
                             hidden_dims=self.cfg.dev_policy_specs['scale_mlp'],
                             activation=self.cfg.dev_policy_specs['scale_htype'])
        cur_dim = self.scale_mlp.out_dim
        self.scale_state_mean = nn.Linear(cur_dim, self.scale_state_dim)
        self.scale_state_mean.weight.data.mul_(1)
        self.scale_state_mean.bias.data.mul_(0.0)
        self.scale_state_log_std = nn.Parameter(
            torch.ones(1, self.scale_state_dim) * self.cfg.dev_policy_specs['scale_log_std'])

        # execution
        if self.cfg.use_entire_obs:
            # use entire obs as control nn input
            self.control_norm = RunningNorm(self.state_dim)
            cur_dim = self.state_dim
        else:
            # use only sim_obs as control nn input
            self.control_norm = RunningNorm(self.sim_obs_dim)
            cur_dim = self.sim_obs_dim

        z_num = cfg.cfg.get('z_num', 6)

        if 'egnn' in cfg.cfg and cfg.cfg['egnn'] and ((self.agent.scope == 'agent0' and 'sg'in cfg.cfg['env_name'].split('-')[-3]) or (self.agent.scope == 'agent1' and 'sg'in cfg.cfg['env_name'].split('-')[-2]) or ('turn'in cfg.cfg['env_name'].split('-')[-5])):
            self.frame_gnn = SGNN(state_dim = cur_dim//len(self.agent.body_ids), attr_fixed_dim = 0, attr_design_dim = 0, msg_dim = 32, p_step = 3, z_num = z_num)
        else:
            self.frame_gnn = None

        self.control_mlp = MLP(cur_dim,
                               hidden_dims=self.cfg.dev_policy_specs['control_mlp'],
                               activation=self.cfg.dev_policy_specs['control_htype'])
        cur_dim = self.control_mlp.out_dim
        self.control_action_mean = nn.Linear(cur_dim, self.sim_action_dim)
        self.control_action_mean.weight.data.mul_(0.1)
        self.control_action_mean.bias.data.mul_(0.0)
        self.control_action_log_std = nn.Parameter(
            torch.ones(1, self.sim_action_dim) * self.cfg.dev_policy_specs['control_log_std'])

        self.is_disc_action = False

        self.fixed_x = None

    def batch_data(self, x):
        stage_ind, scale_state, _, _, sim_obs = zip(*x)
        scale_state = torch.stack(scale_state, 0)
        stage_ind = torch.stack(stage_ind, 0)
        sim_obs = torch.stack(sim_obs, 0)
        return stage_ind, scale_state, sim_obs

    def batch_data_graph(self, x, obs):
        _, _, edges, num_nodes, _ = zip(*x)
        obs= obs.reshape(obs.shape[0]*num_nodes[0], -1)
        # use_transform_action = np.concatenate(use_transform_action)
        if isinstance(num_nodes, np.ndarray):
            num_nodes = torch.tensor(num_nodes, device=obs.device)
        # num_nodes = np.concatenate(num_nodes)
        num_nodes = torch.cat(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        # num_nodes_cum = np.cumsum(num_nodes)
        num_nodes_cum = torch.cumsum(num_nodes,dim=0)
        # body_ind = torch.from_numpy(np.concatenate(body_ind))
        if len(x) > 1:
            repeat_num = [x.shape[1] for x in edges[1:]]
            # e_offset = np.repeat(num_nodes_cum[:-1], repeat_num)
            # e_offset = torch.tensor(e_offset, device=obs.device)
            repeat_num_tensor = torch.tensor(repeat_num, dtype=torch.long,device=obs.device)
            e_offset = torch.repeat_interleave(num_nodes_cum[:-1], repeat_num_tensor)
            edges_new[:, -e_offset.shape[0]:] += e_offset
        return obs, edges_new, num_nodes, num_nodes_cum

    def forward(self, x):
        stages = ['attribute_transform', 'execution']
        x_dict = defaultdict(list)
        design_mask = defaultdict(list)

        for i, x_i in enumerate(x):
            cur_stage = stages[int(x_i[0].item())]
            x_dict[cur_stage].append(x_i)
            for stage in stages:
                design_mask[stage].append(cur_stage == stage)
        for stage in stages:
            design_mask[stage] = torch.BoolTensor(design_mask[stage])
        # print(design_mask)

        if len(x_dict['attribute_transform']) > 0:
            stage_ind, scale_state, sim_obs = self.batch_data(x_dict['attribute_transform'])
            x = scale_state
            x = self.scale_norm(x)
            x = self.scale_mlp(x)
            scale_state_mean = self.scale_state_mean(x)
            # limit the scale vector to [, ]
            # scale_state_mean = torch.ones([1, 8], dtype=torch.float) + scale_state_mean * 0.75

            scale_state_log_std = self.scale_state_log_std.expand_as(scale_state_mean)
            scale_state_std = torch.exp(scale_state_log_std)
            scale_dist = DiagGaussian(scale_state_mean, scale_state_std / 5)
        else:
            scale_dist = None

        if len(x_dict['execution']) > 0:
            stage_ind, scale_state, sim_obs = self.batch_data(x_dict['execution'])
            if self.cfg.use_entire_obs:
                # use entire obs as control nn input
                x = torch.cat((stage_ind, scale_state, sim_obs), -1)
            else:
                # use only sim_obs as control nn input
                x = sim_obs
            x = self.control_norm(x)

            if self.frame_gnn is not None: 
                bz = x.shape[0]
                x, edges, _, num_nodes_cum_control = self.batch_data_graph(x_dict['execution'], x)
                # self.frame_gnn.change_morphology(edges, num_nodes)
                x = self.frame_gnn(x, edges, num_nodes_cum_control,dev=True)
                x = x.reshape(bz, -1)

            x = self.control_mlp(x)
            control_action_mean = self.control_action_mean(x)
            control_action_log_std = self.control_action_log_std.expand_as(control_action_mean)
            control_action_std = torch.exp(control_action_log_std)
            control_dist = DiagGaussian(control_action_mean, control_action_std)
        else:
            control_dist = None

        return scale_dist, control_dist, design_mask, x.device

    def select_action(self, x, mean_action=False):
        """
        :param x: the input is the state of RL
        :return: return the action of RL. The scale vector is listed at first then control action.
        """
        if self.agent.stage == 'attribute_transform':
            self.fixed_x = x
        scale_dist, control_dist, _, _ = self.forward(x)
        if scale_dist is not None:
            scale_action = scale_dist.mean_sample() if mean_action else scale_dist.sample()
            scale = 1
            scale_action = torch.clamp(scale_action, -scale, scale)
        else:
            scale_action = torch.tensor(self.agent.scale_vector)

        if control_dist is not None:
            control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
        else:
            control_action = None

        action = torch.zeros([1, self.action_dim], dtype=torch.float)
        if scale_action is not None:
            action[:, :self.scale_state_dim] = scale_action
        if control_action is not None:
            action[:, self.scale_state_dim:] = control_action

        return action

    def get_log_prob(self, states, actions):
        # print(len(state))
        # print(len(action))
        scale_dist, control_dist, design_mask, device = self.forward(states)
        action_log_prob = torch.zeros(design_mask['execution'].shape[0], 1).to(device)

        # scale transform log prob
        if scale_dist is not None:
            scale_action = []
            for ind, _ in enumerate(actions):
                if design_mask['attribute_transform'][ind]:
                    scale_action.append(_[:self.scale_state_dim])
            scale_action = torch.stack(scale_action, 0)

            scale_state_log_prob = scale_dist.log_prob(scale_action)
            action_log_prob[design_mask['attribute_transform']] = scale_state_log_prob

        # execution log prob
        if control_dist is not None:
            control_action = []
            for ind, _ in enumerate(actions):
                if design_mask['execution'][ind]:
                    control_action.append(_[self.scale_state_dim:])
            control_action = torch.stack(control_action, 0)

            control_action_log_prob = control_dist.log_prob(control_action)
            action_log_prob[design_mask['execution']] = control_action_log_prob

        return action_log_prob