from gym_compete.new_envs.agents import Ant
from gymnasium.spaces import Box
import numpy as np
import os

from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
from lib.utils import get_single_body_qposaddr,get_graph_fc_edges
from competevo.evo_envs.robot.xml_robot import Robot
from custom.utils.transformation import quaternion_matrix

SCALE_MAX = 0.5

class DevAntTurn(Ant):

    def __init__(self, agent_id, cfg, xml_path=None, n_agents=2):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "dev_ant_body_turn.xml")
        super(DevAntTurn, self).__init__(agent_id, xml_path, n_agents)

        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_path, parser=parser)
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')

        self.cfg = cfg

        self.robot = Robot(cfg.robot_cfg, xml=xml_path)

        self.stage = "attribute_transform"
        self.scale_vector = np.random.uniform(low=-1., high=1., size=20)

        self.GOAL = np.array([0, 0, 0])

    @property
    def flag(self):
        return "dev"

    def set_env(self, env):
        super(Ant, self).set_env(env)

        # dimension definition
        self.scale_state_dim = self.scale_vector.size
        self.sim_obs_dim = self.observation_space.shape[0]
        self.sim_action_dim = self.action_space.shape[0]
        self.stage_state_dim = 1
        
        self.action_dim = self.sim_action_dim + self.scale_state_dim
        self.state_dim = self.stage_state_dim + self.scale_state_dim + self.sim_obs_dim

        # print(self.state_dim, self.action_dim)
            
    def set_design_params(self, action):
        scale_state = action[:self.scale_state_dim]
        self.scale_vector = scale_state
        # print(scale_state)

        design_params = self.scale_vector * SCALE_MAX
        a = design_params + 1.
        b = design_params*0.8 + 1 # for gear only

        def multiply_str(s, m):
            res = [str(float(x) * m) for x in s.split()]
            res_str = ' '.join(res)
            return res_str

        agent_body = self.tree.find('body')
        for body in agent_body.iter('body'):
            cur_name = body.get('name')

            # 1
            if cur_name == "1":
                geom = body.find('geom') #1
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[0])
                    geom.set("fromto", p)

            if cur_name == "11":
                p = body.get("pos")
                p = multiply_str(p, a[0])
                body.set("pos", p)

                geom = body.find('geom') #11
                p = geom.get("size")
                p = multiply_str(p, a[1])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[2])
                    geom.set("fromto", p)

            if cur_name == "111":
                p = body.get("pos")
                p = multiply_str(p, a[2])
                body.set("pos", p)

                geom = body.find('geom') #111
                p = geom.get("size")
                p = multiply_str(p, a[3])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[4])
                    geom.set("fromto", p)

            # 2
            if cur_name == "2":
                geom = body.find('geom') #2
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[5])
                    geom.set("fromto", p)

            if cur_name == "12":
                p = body.get("pos")
                p = multiply_str(p, a[5])
                body.set("pos", p)

                geom = body.find('geom') #12
                p = geom.get("size")
                p = multiply_str(p, a[6])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[7])
                    geom.set("fromto", p)

            if cur_name == "112":
                p = body.get("pos")
                p = multiply_str(p, a[7])
                body.set("pos", p)

                geom = body.find('geom') #112
                p = geom.get("size")
                p = multiply_str(p, a[8])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[9])
                    geom.set("fromto", p)

            # 3
            if cur_name == "3":
                geom = body.find('geom') #3
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[10])
                    geom.set("fromto", p)

            if cur_name == "13":
                p = body.get("pos")
                p = multiply_str(p, a[10])
                body.set("pos", p)

                geom = body.find('geom') #13
                p = geom.get("size")
                p = multiply_str(p, a[11])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[12])
                    geom.set("fromto", p)

            if cur_name == "113":
                p = body.get("pos")
                p = multiply_str(p, a[12])
                body.set("pos", p)

                geom = body.find('geom') #113
                p = geom.get("size")
                p = multiply_str(p, a[13])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[14])
                    geom.set("fromto", p)

            # 4
            if cur_name == "4":
                geom = body.find('geom') #4
                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[15])
                    geom.set("fromto", p)

            if cur_name == "14":
                p = body.get("pos")
                p = multiply_str(p, a[15])
                body.set("pos", p)

                geom = body.find('geom') #14
                p = geom.get("size")
                p = multiply_str(p, a[16])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[17])
                    geom.set("fromto", p)

            if cur_name == "114":
                p = body.get("pos")
                p = multiply_str(p, a[17])
                body.set("pos", p)

                geom = body.find('geom') #114
                p = geom.get("size")
                p = multiply_str(p, a[18])
                geom.set("size", p)

                if geom is not None:
                    p = geom.get("fromto")
                    p = multiply_str(p, a[19])
                    geom.set("fromto", p)

        agent_actuator = self.tree.find('actuator')
        for motor in agent_actuator.iter("motor"):
            cur_name = motor.get("name").split('_')[0]

            if cur_name == "11":
                p = motor.get("gear")
                p = multiply_str(p, b[1])
                motor.set("gear", p)

            if cur_name == "111":
                p = motor.get("gear")
                p = multiply_str(p, b[3])
                motor.set("gear", p)

            if cur_name == "12":
                p = motor.get("gear")
                p = multiply_str(p, b[6])
                motor.set("gear", p)

            if cur_name == "112":
                p = motor.get("gear")
                p = multiply_str(p, b[8])
                motor.set("gear", p)

            if cur_name == "13":
                p = motor.get("gear")
                p = multiply_str(p, b[11])
                motor.set("gear", p)

            if cur_name == "113":
                p = motor.get("gear")
                p = multiply_str(p, b[13])
                motor.set("gear", p)

            if cur_name == "14":
                p = motor.get("gear")
                p = multiply_str(p, b[16])
                motor.set("gear", p)

            if cur_name == "114":
                p = motor.get("gear")
                p = multiply_str(p, b[18])
                motor.set("gear", p)

        # print(etree.tostring(self.tree, pretty_print=True).decode('utf-8'))
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')
        # print(self.cur_xml_str)

    def set_goal(self, goal):
        self.GOAL = goal
        # self.move_left = False
        # if self.get_qpos()[0] > 0:
        #     self.move_left = True

    def quat2euler(self, quat):
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1, 1)
        Y = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = np.arctan2(t3, t4)
        return np.array([X, Y, Z])
    # TODO
    def before_step(self):
        self.env.data.geom_xpos[1] = self.GOAL
        self._xposbefore = self.get_body_com("0")[:2]

        self.dist_before = np.linalg.norm(self.GOAL[:2]-self._xposbefore)

        torso_quat = self.env.data.qpos[3:7]
        self.torso_euler = self.quat2euler(torso_quat)
    # TODO
    def after_step(self, action):
        xposafter = self.get_body_com("0")[:2]
        # forward_reward = (xposafter - self._xposbefore) / self.env.dt
        self.dist_after = np.linalg.norm(self.GOAL[:2]-xposafter)
        dist_reward = (self.dist_before - self.dist_after) / self.env.dt


        heading = np.array([np.cos(self.torso_euler[2]), np.sin(self.torso_euler[2])])
        forward_reward =  np.dot(heading, xposafter - self._xposbefore) / self.env.dt
        # if self.move_left:
        #     forward_reward *= -1
        
        # ctrl_cost = .5 * np.square(action).sum()
        # cfrc_ext = self.get_cfrc_ext()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(cfrc_ext, -1, 1))
        # )
    
        # ctrl_cost = 1e-4 * np.square(action).mean()
        ctrl_cost = 1e-2 * np.square(action).sum()
        contact_cost = 0

        survive = 1.0
        reward = 5*forward_reward+ 10*dist_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_dist'] = dist_reward
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dense'] = reward

        info = reward_info
        info['use_transform_action'] = False
        info['stage'] = 'execution'
        info['dist'] = self.dist_after

        # terminate condition
        qpos = self.get_qpos()
        height = qpos[2]
        zdir = quaternion_matrix(qpos[3:7])[:3, 2]
        ang = np.arccos(zdir[2])
        done_condition = self.cfg.done_condition
        min_height = done_condition.get('min_height', 0.28)
        max_height = done_condition.get('max_height', .8) #0.8
        max_ang = done_condition.get('max_ang', 3600)

        # terminated = not (np.isfinite(self.get_qpos()).all() and np.isfinite(self.get_qvel()).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
        terminated = not (np.isfinite(self.get_qpos()).all() and np.isfinite(self.get_qvel()).all() and (height > min_height) and (height < max_height))
        info['dead'] = (height <= min_height) 

        if self.reached_goal():
            # print("setp——reached_goal")
            reward += 1000
            goal_pos_r = np.random.uniform(3, 4)
            if self.symmetric:
                goal_pos_theta = np.random.uniform(0, 2*np.pi)
            else:
                goal_pos_theta = 0
                xposafter[1] = 0
            goal = np.array([goal_pos_r * np.cos(goal_pos_theta), goal_pos_r * np.sin(goal_pos_theta), 0])
            goal[:2] += xposafter
            # print("before")
            # print(self.env.data.geom_xpos[1])
            self.set_goal(goal)
            # print("after")
            # print(self.env.data.geom_xpos[1])

        return reward, terminated, info


    def if_use_transform_action(self):
        return ['attribute_transform', 'execution'].index(self.stage)

    def _get_obs(self, stage=None):
        '''
        Return agent's observations
        '''
        # update stage tag from env
        if stage not in ['attribute_transform', 'execution']:
            stage = 'attribute_transform'
        self.stage = stage

        qpos = self.get_qpos()
        qvel = self.get_qvel()

        root_pos = qpos[:2]
        root_pos = np.append(root_pos, 0)
        other_pos = self.GOAL
        other_pos = other_pos - root_pos

        obs = []
        idx = 0
        agent_body = self.tree.find('body')
        for body in agent_body.iter('body'):
            cur_name = body.get('name')
            if cur_name == "0":
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_pos, np.array([0,0,9.8]), qvel[:6], self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],np.zeros(2)]
            else:
                qs, qe = get_single_body_qposaddr(self.env.model, self.scope + "/" + cur_name)
                if qe - qs >= 1:
                    assert qe - qs == 1
                    angle =  np.append(self.env.data.qpos[qs:qe], self.env.data.qvel[qs-1-self.id:qe-1-self.id])
                else:
                    angle = np.zeros(2)
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_pos, np.array([0,0,9.8]), np.zeros(6), self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],angle]


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


    def set_observation_space(self):
        obs = self._get_obs(self.stage)[-1]
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        # if self.n_agents == 1: return False
        # xpos = self.get_body_com('torso')[0]
        # if self.GOAL > 0 and xpos > self.GOAL:
        #     return True
        # elif self.GOAL < 0 and xpos < self.GOAL:
        #     return True
        if self.dist_after < 0.5:
            return True
        return False


    def reset_agent(self,**kwargs):
        goal_pos_r = np.random.uniform(3, 4)
        if 'symmetric' in kwargs and kwargs['symmetric']:
            goal_pos_theta = np.random.uniform(0, 2*np.pi)
            self.symmetric = True
        else:
            goal_pos_theta = 0
            self.symmetric = False
        goal = np.array([goal_pos_r * np.cos(goal_pos_theta), goal_pos_r * np.sin(goal_pos_theta), 0])
        self.set_goal(goal)

        self.stage = 'attribute_transform'
        self.scale_vector = np.random.uniform(low=-1., high=1., size=20)
