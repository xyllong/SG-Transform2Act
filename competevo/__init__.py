from gymnasium.envs.registration import register
import os

register(
    id='robo-sumo-ants-v0',
    entry_point='competevo.evo_envs:RoboSumoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant_fighter', 'ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgant-sgant-v0',
    entry_point='competevo.evo_envs:RoboSumoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_ant_fighter', 'sg_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgant-ant-v0',
    entry_point='competevo.evo_envs:RoboSumoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_ant_fighter', 'ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='test-robo-sumo-sgant-ant-v0',
    entry_point='competevo.evo_envs:RoboSumoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['test_sg_ant_fighter', 'test_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)


register(
    id='robo-sumo-devants-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_ant_fighter', 'dev_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-evoants-v0',
    entry_point='competevo.evo_envs:RoboSumoEvoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_fighter', 'evo_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.3), (1, 0, 1.3)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-devbugs-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_bug_fighter', 'dev_bug_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevbug-devbug-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_bug_fighter', 'dev_bug_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#devbug-sgdevbug顺序不对，实际是红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevbugs-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_bug_fighter', 'sg_dev_bug_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#devbug-sgdevbug顺序不对，实际是红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevant-devant-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_ant_fighter', 'dev_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevant-sgant-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_ant_fighter', 'sg_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='test-robo-sumo-sgdevant-devant-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['test_sg_dev_ant_fighter', 'test_dev_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevant-sgdevant-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_ant_fighter', 'sg_dev_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],#红队sg 蓝队dev
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)



register(
    id='robo-sumo-devspiders-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_spider_fighter', 'dev_spider_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-sgdevspider-sgdevspider-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['sg_dev_spider_fighter', 'sg_dev_spider_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='test-robo-sumo-sgdevspider-devspider-v0',
    entry_point='competevo.evo_envs:RoboSumoDevEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['test_sg_dev_spider_fighter', 'test_dev_spider_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 4.5,
            },
)

register(
    id='robo-sumo-animals-v0',
    entry_point='competevo.evo_envs:RoboSumoAnimalEnv',
    disable_env_checker=True,
    kwargs={
            'agent_names': ['bug_fighter', 'dev_spider_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            # 'world_xml_path': "../../competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            'min_radius': 2.5,
            'max_radius': 3.5,
            },
)

register(
    id='run-to-goal-evoants-v0',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant', 'evo_ant'],
            'init_pos': [(-2, 0, 0.75), (2, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-devants-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_ant', 'dev_ant'],
            'init_pos': [(-1, 0, 0.75), (1, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-evoant-v0',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v0',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-devant-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_ant_turn'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-sgdevant-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_ant_turn'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-ant-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant_turn'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-sgant-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant_turn'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-devhuman-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_humanoid'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 1.4)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-sgdevhuman-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_humanoid'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 1.4)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-human-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['humanoid'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 1.4)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-sghuman-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['humanoid'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 1.4)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v1',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn1'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v2',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn2'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v3',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn3'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v4',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn4'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='turn-to-goal-evoant-v5',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_turn5'],
            # 'init_pos': [(-2, 0, 0.75)],
            # 'ini_euler': [(0, 0, 0)],
            'init_pos': [(0, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'rgb': [(0.98, 0.87, 0.67)],
            'max_episode_steps': 1000,
            },
)

register(
    id='run-to-goal-devbugs-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_bug', 'dev_bug'],
            'init_pos': [(-1, 0, 0.75), (1, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            # 'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-devspiders-v0',
    entry_point='competevo.evo_envs:MultiDevAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_spider', 'dev_spider'],
            'init_pos': [(-1, 0, 0.75), (1, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-animals-v0',
    entry_point='competevo.evo_envs:MultiAnimalEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['dev_bug', 'dev_bug'],
            'init_pos': [(-1, 0, 0.75), (1, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            # 'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'rgb': [(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
            'max_episode_steps': 500,
            },
)