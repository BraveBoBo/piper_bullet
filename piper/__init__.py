from gym.envs.registration import register


register(id='RigidGrasping-v1', entry_point='envs:RigidGrasping', order_enforce=False,disable_env_checker=True)
