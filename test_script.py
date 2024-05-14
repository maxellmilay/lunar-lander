from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

environment_name = 'LunarLander-v2'
vec_env = make_vec_env(environment_name, n_envs=4)

model = PPO.load("lunar_lander.keras", env=vec_env)

obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")