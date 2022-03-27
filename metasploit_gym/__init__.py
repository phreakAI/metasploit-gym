from gym.envs.registration import register

from .metasploit_env import MetasploitNetworkEnv, MetasploitSimulatorEnv

environments = [["MetasploitNetworkEnv", "v0"], ["MetasploitSimulatorEnv", "v0"]]


for environment in environments:
    register(
        id=f"{environment[0]}-{environment[1]}",
        entry_point=f"metasploit_gym:{environment}",
        nondeterministic=True,
    )
