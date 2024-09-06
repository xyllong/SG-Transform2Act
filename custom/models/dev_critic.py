"""
    This uses only one policy to learn scale vector and control.
"""
import numpy as np
import torch.nn as nn
import torch
from lib.rl.core.running_norm import RunningNorm
from lib.models.mlp import MLP

from custom.models.sgnn import SGNN

class DevValue(nn.Module):
    def __init__(self, cfg, agent):
        super(DevValue, self).__init__()
        self.cfg = cfg
        self.agent = agent
        # dimension define
        self.scale_state_dim = agent.scale_state_dim
        self.sim_action_dim = agent.sim_action_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.action_dim = agent.action_dim
        self.state_dim = agent.state_dim

        self.norm = RunningNorm(self.state_dim)
        cur_dim = self.state_dim


        if 'egnn' in cfg.cfg and cfg.cfg['egnn'] and ((self.agent.scope == 'agent0' and 'sg'in cfg.cfg['env_name'].split('-')[-3]) or (self.agent.scope == 'agent1' and 'sg'in cfg.cfg['env_name'].split('-')[-2]) or ('turn'in cfg.cfg['env_name'].split('-')[-5])):
            self.frame_gnn = SGNN(state_dim = self.sim_obs_dim//len(self.agent.body_ids), attr_fixed_dim = 0, attr_design_dim = 0, msg_dim = 32, p_step = 3, z_num = 7- ('turn'in cfg.cfg['env_name'].split('-')[-5]))
        else:
            self.frame_gnn = None

        self.mlp = MLP(cur_dim,
                       hidden_dims=self.cfg.dev_value_specs['mlp'],
                       activation=self.cfg.dev_value_specs['htype'])
        cur_dim = self.mlp.out_dim
        self.value_head = nn.Linear(cur_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def batch_data(self, x):
        stage_ind, scale_state, _, _,  sim_obs = zip(*x)
        scale_state = torch.stack(scale_state, 0)
        stage_ind = torch.stack(stage_ind, 0)
        sim_obs = torch.stack(sim_obs, 0)
        return stage_ind, scale_state, sim_obs
    
    def batch_data_graph(self, x, obs):
        _, _, edges, num_nodes, _ = zip(*x)
        obs= obs.reshape(obs.shape[0]*num_nodes[0], -1)
        # use_transform_action = np.concatenate(use_transform_action)
        num_nodes = torch.cat(num_nodes)
        edges_new = torch.cat(edges, dim=1)
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

    def forward(self, x_dict):
        stage_ind, scale_state, sim_obs = self.batch_data(x_dict)
        if self.frame_gnn is not None: 
            bz = sim_obs.shape[0]
            sim_obs, edges, _, num_nodes_cum_control = self.batch_data_graph(x_dict, sim_obs)
            # self.frame_gnn.change_morphology(edges, num_nodes)
            sim_obs = self.frame_gnn(sim_obs, edges, num_nodes_cum_control,dev=True)
            sim_obs = sim_obs.reshape(bz, -1)
        x = torch.cat((stage_ind, scale_state, sim_obs), -1)
        x = self.norm(x)

        x = self.mlp(x)
        value = self.value_head(x)
        return value