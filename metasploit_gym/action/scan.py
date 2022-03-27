"""Scanning actions
"""
from .action import Scan
from pymetasploit3.msfconsole import MsfRpcConsole
from pymetasploit3.msfrpc import MsfRpcMethod
import time


class PortScan(Scan):
    """The port scan will be our only scan, at first.
    It will represent the biggest, heaviest, and dumbest scan metasploit has.
    All ports, all services, looking for version numbers. It will be capable of giving us a "full starting state"
    of any machine we currently have subnet access to.
    """

    def __init__(self, target=(0, 0)):
        self.name = "PortScan"
        self.target = target
        self.req_access = None
        self.req_os = None
        self.req_version = None
        super(Scan, self).__init__(
            self.name, self.target, self.req_access, self.req_os, self.req_version
        )

    def execute(self, client, host):
        """
        Using the metasploit client, perform the action required
        TODO: Hold until you can confirm the action is completed to allow the calling function to observe state
        :param client: a metasploit client that can have the action run
        :return:
        """
        c_id = client.call(MsfRpcMethod.ConsoleCreate)["id"]
        client.consoles.console(c_id).write(f"db_nmap {host}\n")
        out = client.consoles.console(c_id).read()["data"]
        timeout = 150
        counter = 0
        while counter < timeout:
            out += client.consoles.console(c_id).read()["data"]
            if "Nmap done" in out:
                break
            time.sleep(1)
            counter += 1
