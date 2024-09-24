from gym_compete.new_envs.agents import *
from gymnasium.spaces import Box
import numpy as np
import os

from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from lib.utils import get_single_body_qposaddr,get_graph_fc_edges
from competevo.evo_envs.robot.xml_robot import Robot

SCALE_MAX = 0.5

class AntFighter(RoboAntFighter):
    CFRC_CLIP = 100.

    COST_COEFS = {
        'ctrl': 1e-1,
        # 'pain': 1e-4,
        # 'attack': 1e-1,
    }

    JNT_NPOS = {
        0: 7,
        1: 4,
        2: 1,
        3: 1,
    }

    def __init__(self, agent_id, cfg, xml_path=None, n_agents=2):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "dev_ant_body.xml")
        super(AntFighter, self).__init__(agent_id, xml_path, n_agents)

        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_path, parser=parser)
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')

        self.cfg = cfg

        self.robot = Robot(cfg.robot_cfg, xml=xml_path)

        self.stage = "execution"
        self.scale_vector = np.zeros(20)
    
    @property
    def flag(self):
        return "dev"

    def set_env(self, env):
        super(RoboAntFighter, self).set_env(env)
        self.arena_id = self.env.geom_names.index('arena')
        self.arena_height = self.env.model.geom_size[self.arena_id][1] * 2

        # dimension definition
        self.scale_state_dim = self.scale_vector.size
        self.sim_obs_dim = self.observation_space.shape[0]
        self.sim_action_dim = self.action_space.shape[0]
        self.stage_state_dim = 1
        
        self.action_dim = self.sim_action_dim + self.scale_state_dim
        self.state_dim = self.stage_state_dim + self.scale_state_dim + self.sim_obs_dim

        # print(self.state_dim, self.action_dim)

    def set_design_params(self, action):
        pass

    def before_step(self):
        self.posbefore = self.get_qpos()[:2].copy()
    
    def after_step(self, action):
        """ RoboSumo design.
        """
        self.posafter = self.get_qpos()[:2].copy()
        # Control cost
        control_reward = - self.COST_COEFS['ctrl'] * np.square(action).sum()

        alive_reward = 2.0

        return control_reward, alive_reward

    def if_use_transform_action(self):
        return ['attribute_transform', 'execution'].index(self.stage)

    def _get_obs(self, stage=None):
        '''
        Return agent's observations
        '''
        # update stage tag from env
        self.stage = 'execution'

        # Observe self
        self_forces = np.abs(np.clip(
            self.get_cfrc_ext(), -self.CFRC_CLIP, self.CFRC_CLIP))
        # obs  = [
        #     self.get_qpos().flat,           # self all positions
        #     self.get_qvel().flat,           # self all velocities
        #     self_forces.flat,               # self all forces
        # ]
        
        # Observe opponents
        other_qpos = self.get_other_qpos()
        # if other_qpos.shape == (0,):
        #     # other_qpos = np.zeros(2) # x and y
        #     other_qpos = np.random.uniform(-5, 5, 2)

        # obs.extend([
        #     other_qpos[:2].flat,    # opponent torso position
        # ])

        # torso_xmat = self.get_torso_xmat()
        # # print(torso_xmat)
        # obs.extend([
        #     torso_xmat.flat,
        # ])

        qpos = self.get_qpos()
        qvel = self.get_qvel()
        root_pos = np.array([0,0,0])
        obs = []
        idx = 0
        agent_body = self.tree.find('body')
        for body in agent_body.iter('body'):
            cur_name = body.get('name')
            # forces = [self_forces[id] for id in self_forces_id[idx]]
            if cur_name == "0":
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_qpos, np.array([0,0,9.8]),  self_forces[idx], qvel[:6],self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],np.zeros(2)]
            else:
                qs, qe = get_single_body_qposaddr(self.env.model, self.scope + "/" + cur_name)
                if qe - qs >= 1:
                    assert qe - qs == 1
                    angle =  np.append(self.env.data.qpos[qs:qe], self.env.data.qvel[qs-1-self.id:qe-1-self.id])
                else:
                    angle = np.zeros(2)
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_qpos,  np.array([0,0,9.8]),  self_forces[idx], np.zeros(6),self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],angle]


            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
            idx += 1


        num_nodes = np.array([len(obs)])
        sim_obs = np.concatenate(obs)
        assert np.isfinite(sim_obs).all(), "Ant observation is not finite!!"

        edges = self.robot.get_gnn_edges()
        # edges = get_graph_fc_edges(num_nodes[0])

        obs = [np.array([self.if_use_transform_action()]), self.scale_vector, edges, num_nodes, sim_obs]

        return obs

    def get_torso_xmat(self):
        return self.env.data.xmat[self.body_ids[self.body_names.index('agent%d/0' % self.id)]]

    def set_observation_space(self):
        obs = self._get_obs(self.stage)[-1]
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)
