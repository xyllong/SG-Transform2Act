from gym_compete.new_envs.agents.agent import Agent
from gymnasium.spaces import Box
import numpy as np
import os

from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO

from competevo.evo_envs.robot.xml_robot import Robot
from lib.utils import get_single_body_qposaddr, get_graph_fc_edges
from custom.utils.transformation import quaternion_matrix

SCALE_MAX = 0.3

def mass_center(mass, xpos):
    # This is a bug of openai multiagent-competition release.
    # It should keep the same dimension when array broadcast.
    # Modified by git @KJaebye.
    mass = mass[:, np.newaxis]
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class DevHumanoid(Agent):

    def __init__(self, agent_id, cfg, xml_path=None, n_agents=2, **kwargs):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml")
        super(DevHumanoid, self).__init__(agent_id, xml_path, n_agents, **kwargs)

        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_path, parser=parser)
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')

        self.cfg = cfg
        
        self.robot = Robot(cfg.robot_cfg, xml=xml_path)

        self.stage = "attribute_transform"
        self.scale_vector = np.random.uniform(low=-1., high=1., size=20)

    @property
    def flag(self):
        return "dev"
        
    def set_env(self, env):
        self.env = env
        self._env_init = True
        self._set_body()
        self._set_joint()
        if self.n_agents > 1:
            self._set_other_joint()
        self.set_observation_space()
        self.set_action_space()

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
        b = design_params*0.5 + 1 # for gear only

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

    def before_step(self):
        self.env.data.geom_xpos[1] = self.GOAL
        self._pos_before = mass_center(self.get_body_mass(), self.get_xipos())[:2]

        torso_quat = self.env.data.qpos[3:7]
        self.torso_euler = self.quat2euler(torso_quat)
        #degree
        # self.torso_euler = np.rad2deg(self.torso_euler)
        # print("Torso Euler:", self.torso_euler)
        # self.heading = [np.cos(self.torso_euler[2]), np.sin(self.torso_euler[2])]
        self.dist_before = np.linalg.norm(self.GOAL[:2]-self._pos_before)

    def after_step(self, action):
        pos_after = mass_center(self.get_body_mass(), self.get_xipos())[:2]
        # forward_reward = 1.25 * (pos_after - self._pos_before) / self.env.model.opt.timestep
        # forward_reward = .25 * (pos_after - self._pos_before) / self.env.model.opt.timestep
        heading = np.array([np.cos(self.torso_euler[2]), np.sin(self.torso_euler[2])])
        forward_reward = 1.25 * np.dot(heading, pos_after - self._pos_before) / self.env.model.opt.timestep

        self.dist_after = np.linalg.norm(self.GOAL[:2]-pos_after)
        dist_reward = (self.dist_before - self.dist_after) / self.env.model.opt.timestep

        # if self.move_left:
        #     forward_reward *= -1
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        survive = 5.0
        reward = forward_reward - ctrl_cost - contact_cost + survive + dist_reward

        # reward_goal = - np.abs(qpos[0].item() - self.GOAL)
        # reward += reward_goal

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dist'] = dist_reward
        # if self.team == 'walker':
        #     reward_info['reward_goal_dist'] = reward_goal
        reward_info['reward_dense'] = reward
        
        reward_info['dist'] = self.dist_after

        # done = not agent_standing
        terminated = bool(qpos[2] < 1. or qpos[2] > 2.)

        if self.reached_goal():
            # print("setp——reached_goal")
            reward += 1000
            goal_pos_r = np.random.uniform(3, 4)
            if self.symmetric:
                goal_pos_theta = np.random.uniform(0, 2*np.pi)
            else:
                goal_pos_theta = 0
                pos_after[1] = 0
            goal = np.array([goal_pos_r * np.cos(goal_pos_theta), goal_pos_r * np.sin(goal_pos_theta), 0])
            goal[:2] += pos_after
            # print("before")
            # print(self.env.data.geom_xpos[1])
            self.set_goal(goal)
            # print("after")
            # print(self.env.data.geom_xpos[1])

        return reward, terminated, reward_info

    def if_use_transform_action(self):
        return ['attribute_transform', 'execution'].index(self.stage)

    # def _get_obs(self, stage=None):
    #     '''
    #     Return agent's observations
    #     '''
    #     # update stage tag from env
    #     if stage not in ['attribute_transform', 'execution']:
    #         stage = 'attribute_transform'
    #     self.stage = stage

    #     my_pos = self.get_qpos()
    #     other_pos = self.get_other_qpos()
    #     vel = self.get_qvel()
    #     cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
    #     cvel = self.get_cvel()
    #     cinert = self.get_cinert()
    #     qfrc_actuator = self.get_qfrc_actuator()

    #     sim_obs = np.concatenate(
    #         [my_pos.flat, vel.flat,
    #          cinert.flat, cvel.flat,
    #          qfrc_actuator.flat, cfrc_ext.flat,
    #          other_pos.flat]
    #     )
    #     assert np.isfinite(sim_obs).all(), "Humanoid observation is not finite!!"
        
    #     obs = [np.array([self.if_use_transform_action()]), self.scale_vector, sim_obs]

    #     return obs
    
    def _get_obs(self, stage=None):
        '''
        Return agent's observations
        '''
        # update stage tag from env
        if stage not in ['attribute_transform', 'execution']:
            stage = 'attribute_transform'
        self.stage = stage

        # Observe self
        # self_forces = np.abs(np.clip(
        #     self.get_cfrc_ext(), -self.CFRC_CLIP, self.CFRC_CLIP))
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
            if cur_name == "torso":
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_qpos, np.array([0,0,9.8]), qvel[:6],self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],np.zeros(6)]
            else:
                qs, qe = get_single_body_qposaddr(self.env.model, self.scope + "/" + cur_name)
                if qe - qs >= 1:
                    # assert qe - qs == 1
                    angle0 = self.env.data.qpos[qs:qe]
                    #我要把angle0填充到3维
                    angle0 = np.append(angle0, np.zeros(3-len(angle0)))
                    angle1 = self.env.data.qvel[qs-1-self.id:qe-1-self.id]
                    angle1 = np.append(angle1, np.zeros(3-len(angle1)))
                    angle =  np.append(angle0, angle1)
                else:
                    angle = np.zeros(6)
                obs_i = [self.env.data.body(self.scope + "/" +cur_name).xipos - root_pos, other_qpos,  np.array([0,0,9.8]),  np.zeros(6),self.env.data.body(self.scope + "/" +cur_name).xipos[2:3],angle]


            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
            idx += 1


        num_nodes = np.array([len(obs)])
        sim_obs = np.concatenate(obs)
        assert np.isfinite(sim_obs).all(), "observation is not finite!!"

        edges = self.robot.get_gnn_edges()
        # edges = get_graph_fc_edges(num_nodes[0])

        obs = [np.array([self.if_use_transform_action()]), self.scale_vector, edges, num_nodes, sim_obs]

        return obs

    # def _get_obs_relative(self):
    #     '''
    #     Return agent's observations, positions are relative
    #     '''
    #     qpos = self.get_qpos()
    #     my_pos = qpos[2:]
    #     other_agents_qpos = self.get_other_agent_qpos()
    #     all_other_qpos = []
    #     for i in range(self.n_agents):
    #         if i == self.id: continue
    #         other_qpos = other_agents_qpos[i]
    #         other_relative_xy = other_qpos[:2] - qpos[:2]
    #         other_qpos = np.concatenate([other_relative_xy.flat, other_qpos[2:].flat], axis=0)
    #         all_other_qpos.append(other_qpos)
    #     all_other_qpos = np.concatenate(all_other_qpos)

    #     vel = self.get_qvel()
    #     cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
    #     cvel = self.get_cvel()
    #     cinert = self.get_cinert()
    #     qfrc_actuator = self.get_qfrc_actuator()

    #     obs = np.concatenate(
    #         [my_pos.flat, vel.flat,
    #          cinert.flat, cvel.flat,
    #          qfrc_actuator.flat, cfrc_ext.flat,
    #          all_other_qpos.flat]
    #     )
    #     assert np.isfinite(obs).all(), "Humanoid observation is not finite!!"
    #     return obs

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
