import pytest
from metasploit_gym.metasploit_env import MetasploitNetworkEnv
from dotenv import load_dotenv
import os

load_dotenv()
METASPLOIT_PASS = os.getenv("METASPLOIT_PASSWORD", default=None)
METASPLOIT_PORT = os.getenv("METASPLOIT_PORT", default=None)
METASPLOIT_HOST = os.getenv("METASPLOIT_HOST", default=None)


@pytest.fixture()
def network_env():
    network_env = MetasploitNetworkEnv(
        msf_host=METASPLOIT_HOST,
        msf_rpc_password=METASPLOIT_PASS,
        msf_rpc_port=METASPLOIT_PORT,
    )  # maybe host config as argument here
    yield network_env
    network_env.reset()
