import pytest
from metasploit_gym.metasploit_env import MetasploitNetworkEnv
from metasploit_gym.action.scan import PortScan
from metasploit_gym.action.exploit import (
    SSH_Bruteforce,
    FTP_Bruteforce,
    SMB_Bruteforce,
    Telnet_Bruteforce,
    VSFTPD,
    JavaRMIServer,
    Ms08_067_Netapi,
    ManageEngine_Auth_Upload,
    ApacheJamesExecution,
)
from pymetasploit3.msfrpc import MsfRpcClient
from pymetasploit3.msfrpc import MsfRpcMethod
import os
from dotenv import load_dotenv

load_dotenv()
METASPLOIT_PASS = os.getenv("METASPLOIT_PASSWORD", default=None)
METASPLOIT_PORT = os.getenv("METASPLOIT_PORT", default=None)
METASPLOIT_HOST = os.getenv("METASPLOIT_HOST", default=None)

if METASPLOIT_HOST is None or METASPLOIT_PORT is None or METASPLOIT_PASS is None:
    raise ValueError(
        "Please include a .env file with METASPLOIT_PASS, METASPLOIT_PORT, and METASPLOIT HOST set to the values of the msgrpc service"
    )


@pytest.fixture()
def client():
    client = MsfRpcClient(
        METASPLOIT_PASS, server=METASPLOIT_HOST, port=METASPLOIT_PORT, ssl=True
    )
    yield client
    client.call(MsfRpcMethod.AuthLogout)


@pytest.fixture()
def network_env():
    network_env = MetasploitNetworkEnv(reset_function=None)  # maybe host config as argument here
    yield network_env
    network_env.reset()


def test_connection(client):
    """
    Test whether we cannot connect to the metasploit server
    """
    assert [m for m in dir(client) if not m.startswith("_")] != []


def test_db_connection(client):
    """
    Test whether there's a persistent connection to the database
    :return:
    """
    default_workspace_hosts = client.db.workspaces.list[0]
    assert default_workspace_hosts["name"] == "metasploitgym"


def test_network_scan(network_env):
    """
    Test scanning in the real network and assume port 22 is open
    """
    action = PortScan()
    network_env.step(action)
    assert 1 == 1


def test_ssh_scan(network_env):
    """
    Test ssh module execution
    """
    action = SSH_Bruteforce()
    network_env.step(action)
    assert 1 == 1


def test_ftp_scan(network_env):
    """
    test ftp module execution
    """
    action = FTP_Bruteforce()
    network_env.step(action)
    assert 1 == 1


def test_smb_scan(network_env):
    action = SMB_Bruteforce()
    network_env.step(action)
    assert 1 == 1


def test_telnet_scan(network_env):
    action = Telnet_Bruteforce()
    network_env.step(action)
    assert 1 == 1


def test_vsftpd_exploit(network_env):
    action = VSFTPD()
    network_env.step(action)
    assert 1 == 1


def test_java_rmi_server(network_env):
    action = JavaRMIServer()
    network_env.step(action)
    assert 1 == 1


def test_ms08_067_netapi(network_env):
    action = Ms08_067_Netapi()
    network_env.step(action)
    assert 1 == 1


def test_manageengine_auth_upload(network_env):
    action = ManageEngine_Auth_Upload()
    network_env.step(action)
    assert 1 == 1


def test_apache_james_auth_upload(network_env):
    action = ApacheJamesExecution()
    network_env.step(action)
    assert 1 == 1


def test_env_update(network_env):
    network_env.update_env()
    print(network_env.network.vectorize())


def test_env_reset(network_env):
    action = ApacheJamesExecution()
    network_env.step(action)
    network_env.update_env()
    network_env.reset()
