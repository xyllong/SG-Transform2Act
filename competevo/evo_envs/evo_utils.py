import xml.etree.ElementTree as ET
import colorsys
import numpy as np
import os

def list_filter(lambda_fn, iterable):
    return list(filter(lambda_fn, iterable))

def get_distinct_colors(n=2):
    '''
    taken from: https://stackoverflow.com/a/876872
    '''
    HSV_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def set_class(root, prop, agent_class):
    if root is None:
        return
    # root_class = root.get('class')
    if root.tag == prop:
        root.set('class', agent_class)
    children = list(root)
    for child in children:
        set_class(child, prop, agent_class)

def set_geom_class(root, name):
    set_class(root, 'geom', name)

def set_motor_class(root, name):
    set_class(root, 'motor', name)

def add_prefix(root, prop, prefix, force_set=False):
    if root is None:
        return
    root_prop_val = root.get(prop)
    if root_prop_val is not None:
        root.set(prop, prefix + '/' + root_prop_val)
    elif force_set:
        root.set(prop, prefix + '/' + 'anon' + str(np.random.randint(1, 1e10)))
    children = list(root)
    for child in children:
        add_prefix(child, prop, prefix, force_set)


def tuple_to_str(tp):
    return " ".join(map(str, tp))


def create_multiagent_xml_str(
        world_xml, 
        all_agent_xml_strs, 
        agent_scopes=None,
        ini_pos=None, 
        ini_euler=None,
        rgb=None,
        symmetric=False,
    ):
    world = ET.parse(world_xml)
    world_root = world.getroot()
    world_default = world_root.find('default')
    world_body = world_root.find('worldbody')
    # goal_pos_r = np.random.uniform(5, 10)
    # goal_pos_theta = np.random.uniform(0, 2*np.pi)
    # if symmetric:
    #     goal_pos = (goal_pos_r * np.cos(goal_pos_theta), goal_pos_r * np.sin(goal_pos_theta), 0)
    # else:
    #     goal_pos = (-10,0,0)
    # for child in list(world_body):
    #     if child.tag == 'geom' and child.get('name') == 'goal':
    #         child.set('pos', tuple_to_str(goal_pos))
        # elif child.tag == 'geom' and child.get('name') == 'rightgoal':
        #     child.set('visible', 'False')
        # elif child.tag == 'geom' and child.get('name') == 'leftgoal':
        #     child.set('visible', 'False')

    world_actuator = None
    world_tendons = None
    n_agents = len(all_agent_xml_strs)
    if rgb is None:
        rgb = get_distinct_colors(n_agents)
    RGB_tuples = list(
        map(lambda x: tuple_to_str(x), rgb)
    )
    if agent_scopes is None:
        agent_scopes = ['agent' + str(i) for i in range(n_agents)]

    if ini_pos is None:
        ini_pos = [(-i, 0, 0.75) for i in np.linspace(-n_agents, n_agents, n_agents)]
    # ini_pos = list(map(lambda x: tuple_to_str(x), ini_pos))

    for i in range(n_agents):
        agent_default = ET.SubElement(
            world_default, 'default', attrib={'class': agent_scopes[i]}
        )
        rgba = RGB_tuples[i] + " 1"
        agent_xml = ET.fromstring(all_agent_xml_strs[i])
        # print(ET.tostring(agent_xml.getroot(), encoding='utf-8', method='xml').decode('utf-8'))
        default = agent_xml.find('default')
        for child in list(default):
            if child.tag == 'geom':
                child.set('rgba', rgba)
                child.set("conaffinity", str(i))
                child.set("contype", str(1-i))
            agent_default.append(child)

        agent_body = agent_xml.find('body')
        if agent_body.get('pos'):
            oripos = list(map(float, agent_body.get('pos').strip().split(" ")))
            # keep original y and z coordinates

            # if symmetric and n_agents>1:
            #     goal_pos_r = np.random.uniform(1, 4)
            #     goal_pos_theta = np.random.uniform(0, 2*np.pi)
            #     pos =list((goal_pos_r * np.cos(goal_pos_theta), goal_pos_r * np.sin(goal_pos_theta), ini_pos[i][2]))
            # else:
            pos = list(ini_pos[i])

            
            # pos[1] = oripos[1]
            # pos[2] = oripos[2]
            # print(tuple_to_str(pos))
            agent_body.set('pos', tuple_to_str(pos))
        if agent_body.get('euler'):
            orieuler = list(map(float, agent_body.get('euler').strip().split(" ")))
            # keep original y and z coordinates

            # if symmetric:
            yaw = np.random.uniform(low=-np.pi, high=np.pi)/np.pi*180
            euler = list((0, 0, yaw))
            # else:
            #     euler = list(ini_euler[i])

            # euler[1] = orieuler[1]
            # euler[2] = orieuler[2]
            # print(tuple_to_str(euler))
            agent_body.set('euler', tuple_to_str(euler))
        # add class to all geoms
        set_geom_class(agent_body, agent_scopes[i])
        # add prefix to all names, important to map joints
        add_prefix(agent_body, 'name', agent_scopes[i], force_set=True)
        # add aggent body to xml
        world_body.append(agent_body)
        # get agent actuators
        agent_actuator = agent_xml.find('actuator')
        # add same prefix to all motor joints
        add_prefix(agent_actuator, 'joint', agent_scopes[i])
        add_prefix(agent_actuator, 'name', agent_scopes[i])
        # add actuator
        set_motor_class(agent_actuator, agent_scopes[i])
        if world_actuator is None:
            world_root.append(agent_actuator)
            world_actuator = world_root.find('actuator')
            # print(world_actuator)
            # print(ET.tostring(world_root))
        else:
            for motor in list(agent_actuator):
                world_actuator.append(motor)
        # get agent tendons if exists
        agent_tendon = agent_xml.find('tendon')
        if agent_tendon:
            # add same prefix to all motor joints
            add_prefix(agent_tendon, 'joint', agent_scopes[i])
            add_prefix(agent_tendon, 'name', agent_scopes[i])
            # add tendon
            if world_tendons is None:
                world_root.append(agent_tendon)
                world_tendons = world_root.find('tendon')
                # print(world_actuator)
                # print(ET.tostring(world_root))
            else:
                for tendon in list(agent_tendon):
                    world_tendons.append(tendon)

    return ET.tostring(world_root, encoding='utf-8', method='xml').decode('utf-8')

def create_multiagent_xml(
        world_xml, 
        all_agent_xmls, 
        agent_scopes=None,
        outdir=os.path.join(os.path.dirname(__file__), "assets"),
        outpath=None,
        ini_pos=None, 
        ini_euler=None,
        rgb=None
    ):
    world = ET.parse(world_xml)
    world_root = world.getroot()
    world_default = world_root.find('default')
    world_body = world_root.find('worldbody')
    world_actuator = None
    world_tendons = None
    n_agents = len(all_agent_xmls)
    if rgb is None:
        rgb = get_distinct_colors(n_agents)
    RGB_tuples = list(
        map(lambda x: tuple_to_str(x), rgb)
    )
    if agent_scopes is None:
        agent_scopes = ['agent' + str(i) for i in range(n_agents)]

    if ini_pos is None:
        ini_pos = [(-i, 0, 0.75) for i in np.linspace(-n_agents, n_agents, n_agents)]
    # ini_pos = list(map(lambda x: tuple_to_str(x), ini_pos))

    for i in range(n_agents):
        agent_default = ET.SubElement(
            world_default, 'default', attrib={'class': agent_scopes[i]}
        )
        rgba = RGB_tuples[i] + " 1"
        agent_xml = ET.parse(all_agent_xmls[i])
        # print(ET.tostring(agent_xml.getroot(), encoding='utf-8', method='xml').decode('utf-8'))
        default = agent_xml.find('default')
        for child in list(default):
            if child.tag == 'geom':
                child.set('rgba', rgba)
                child.set("conaffinity", str(i))
                child.set("contype", str(1-i))
            agent_default.append(child)

        agent_body = agent_xml.find('body')
        if agent_body.get('pos'):
            oripos = list(map(float, agent_body.get('pos').strip().split(" ")))
            # keep original y and z coordinates
            pos = list(ini_pos[i])
            # pos[1] = oripos[1]
            # pos[2] = oripos[2]
            # print(tuple_to_str(pos))
            agent_body.set('pos', tuple_to_str(pos))
        if agent_body.get('euler'):
            orieuler = list(map(float, agent_body.get('euler').strip().split(" ")))
            # keep original y and z coordinates
            euler = list(ini_euler[i])
            # euler[1] = orieuler[1]
            # euler[2] = orieuler[2]
            # print(tuple_to_str(euler))
            agent_body.set('euler', tuple_to_str(euler))
        # add class to all geoms
        set_geom_class(agent_body, agent_scopes[i])
        # add prefix to all names, important to map joints
        add_prefix(agent_body, 'name', agent_scopes[i], force_set=True)
        # add aggent body to xml
        world_body.append(agent_body)
        # get agent actuators
        agent_actuator = agent_xml.find('actuator')
        # add same prefix to all motor joints
        add_prefix(agent_actuator, 'joint', agent_scopes[i])
        add_prefix(agent_actuator, 'name', agent_scopes[i])
        # add actuator
        set_motor_class(agent_actuator, agent_scopes[i])
        if world_actuator is None:
            world_root.append(agent_actuator)
            world_actuator = world_root.find('actuator')
            # print(world_actuator)
            # print(ET.tostring(world_root))
        else:
            for motor in list(agent_actuator):
                world_actuator.append(motor)
        # get agent tendons if exists
        agent_tendon = agent_xml.find('tendon')
        if agent_tendon:
            # add same prefix to all motor joints
            add_prefix(agent_tendon, 'joint', agent_scopes[i])
            add_prefix(agent_tendon, 'name', agent_scopes[i])
            # add tendon
            if world_tendons is None:
                world_root.append(agent_tendon)
                world_tendons = world_root.find('tendon')
                # print(world_actuator)
                # print(ET.tostring(world_root))
            else:
                for tendon in list(agent_tendon):
                    world_tendons.append(tendon)

    outname = world_xml.split("/")[-1].split(".xml")[0]  + '.' + ".".join(map(lambda x: x.split("/")[-1].split(".xml")[0], all_agent_xmls)) + ".xml"
    outpath = outdir + '/' + outname

    world.write(outpath)
    return ET.tostring(world_root), outpath