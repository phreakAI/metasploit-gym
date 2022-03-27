import numpy as np

from ..action.action import Exploit
from .utils import SERVICES_TCP, REWARDS


class Network:
    """Collect of Hosts objects, with helper functions for comparing them
    """
    def __init__(self, max_subnets, max_hosts_per_subnet):
        self.hosts = []
        self.max_subnets = max_subnets
        self.max_hosts_per_subnet = max_hosts_per_subnet
        self.num_exploits = len(Exploit.__subclasses__())
        self.host_vector_size = self.calculate_host_vector_size()

    def max_reward(self):
        """Returns the maximum reward. Currently this is configured to be gaining root access to a host.
           Configurable in utils.py

        Returns:
            int: The value of 'root' in the Rewards dictionary
        """
        return REWARDS["root"]

    def check_host_exists(self, board_address):
        """Checks if board address represents a new host or one already occupied.

        Args:
            board_address (str): Address

        Returns:
            bool: True if the Network object knows about the machine, False otherwise
        """
        for host in self.hosts:
            if host.board_address == board_address:
                return True
            else:
                return False

    def compare_updated_host(self, updated_host):
        """Pulls clean data from Metasploits database to see if the last action earned 
           any reward for the host

        Args:
            updated_host (Host): New Host object to be compared to the networks current objects

        Raises:
            ValueError: Raised if the host is new and can't be compared to old data

        Returns:
            _type_: _description_
        """
        for host in self.hosts:
            if host.board_address == updated_host.board_address:
                reward = 0
                new_services = updated_host.service_count - host.service_count
                if new_services > 0:
                    reward += new_services * REWARDS["services"]
                new_vulns = updated_host.vuln_count - host.vuln_count
                if new_vulns > 0:
                    reward += new_vulns * REWARDS["vulns"]
                if updated_host.credentialed_access and not host.credentialed_access:
                    # thats new creds
                    reward += REWARDS["creds"]
                new_loot = updated_host.loot_count - host.loot_count
                if new_loot > 0:
                    reward += new_loot * REWARDS["loot"]
                if updated_host.open_console and not host.open_console:
                    reward += REWARDS["shell"]
                if updated_host.meterpreter_shell and not host.meterpreter_shell:
                    reward += REWARDS["meterpreter"]
                return reward
        raise ValueError("Cannot compare updated host, this host is new")

    def update_host(self, updated_host):
        """After reward has been accounted for, updated Host list with new host

        Args:
            updated_host (Host): Most recent Host constructed from Metasploit database
        """
        for i in range(len(self.hosts)):
            if self.hosts[i].board_address == updated_host.board_address:
                self.hosts[i] = updated_host

    def calculate_host_vector_size(self):
        """Quick math to calculate the desired size of our host vector

        Returns:
            int: Size of a flat host vector
        """
        operating_systems = 2
        loot_slots = 1  # only one kind of loot for our purposes
        cred_slots = 1  # only one kind of cred for our purposes
        shell_types = 2  # regular and meterpreter
        privilege_levels = 5
        return (
            self.max_subnets
            + self.max_hosts_per_subnet
            + operating_systems
            + len(SERVICES_TCP)
            + self.num_exploits
            + loot_slots
            + cred_slots
            + shell_types
            + privilege_levels
        )

    def add_host(self, host):
        """Calculate rewards of host

        Args:
            host (Host): Host to have its rewards calculated

        Raises:
            TypeError: Raised if object is not of type Host
            ValueError: Raised if host has already been accounted for

        Returns:
            int: Reward amount of host
        """
        reward = 0
        if not isinstance(host, Host):
            raise TypeError("Cannot add object to network unless it's of type Host")
        else:
            for current_host in self.hosts:
                if current_host.ip_address == host.ip_address:
                    raise ValueError("Host is already added")
            new_services = host.service_count
            if new_services > 0:
                reward += new_services * REWARDS["services"]
            new_vulns = host.vuln_count
            if new_vulns > 0:
                reward += new_vulns * REWARDS["vulns"]
            if host.credentialed_access:
                reward += REWARDS["creds"]
            new_loot = host.loot_count
            if new_loot > 0:
                reward += new_loot * REWARDS["loot"]
            if host.open_console:
                reward += REWARDS["shell"]
            if host.meterpreter_shell:
                reward += REWARDS["meterpreter"]
            self.hosts.append(host)
            return reward

    def vectorize(self):
        if self.hosts == []:
            return np.zeros(
                (
                    self.calculate_host_vector_size(),
                    self.max_subnets * self.max_hosts_per_subnet,
                )
            )
        array_list = [
            self._network_tensor(self.hosts),
            self._os_tensor(self.hosts),
            self._services_tensor(self.hosts),
            self._vulns_tensor(self.hosts),
            self._loot_tensor(self.hosts),
            self._creds_tensor(self.hosts),
            self._shells_tensor(self.hosts),
            self._privilege_tensor(self.hosts),
        ]
        network_vector = np.concatenate(array_list)
        return network_vector

    def _services_tensor(self, hosts):
        """

        :param hosts:
        :return:
        """
        services_list = sorted(hosts[0].services.keys())
        port_count = len(hosts[0].services)
        service_tensor = np.zeros((port_count, len(hosts)))
        for i in range(len(hosts)):
            host = hosts[i]
            for j in range(len(services_list)):
                port = services_list[j]
                if host.services[port]["status"] == True:
                    service_tensor[j, i] = 1.0
        return service_tensor

    def _network_tensor(self, hosts):
        """
        generate network
        """
        # TODO: handle lookups for larger subnet amounts
        network_tensor = np.zeros(
            (self.max_subnets + self.max_hosts_per_subnet, len(hosts))
        )
        return network_tensor

    def _os_tensor(self, hosts):
        os_tensor = np.zeros(
            (2, len(hosts))
        )  # two OSs, first is Windows second is Linux
        for i in range(len(hosts)):
            host = hosts[i]
            os_tensor[1, i] = 1.0  # setup Linux
        return os_tensor

    def _vulns_tensor(self, hosts):
        vulns_tensor = np.zeros((len(Exploit.__subclasses__()), len(hosts)))
        return vulns_tensor

    def _loot_tensor(self, hosts):
        loot_tensor = np.zeros((1, len(hosts)))
        for i in range(len(hosts)):
            host = hosts[i]
            loot_tensor[0, i] = host.loot_count
        return loot_tensor

    def _creds_tensor(self, hosts):
        cred_tensor = np.zeros((1, len(hosts)))
        for i in range(len(hosts)):
            host = hosts[i]
            if host.has_creds:
                cred_tensor[0, i] = 1.0
        return cred_tensor

    def _shells_tensor(self, hosts):
        shell_tensor = np.zeros((2, len(hosts)))
        for i in range(len(hosts)):
            host = hosts[i]
            if host.open_console:
                shell_tensor[0, i] = 1.0
            if host.meterpreter_shell:
                shell_tensor[1, i] = 1.0
        return shell_tensor

    def _privilege_tensor(self, hosts):
        privilege_tensor = np.zeros((5, len(hosts)))
        return privilege_tensor


class Host:
    """A single host in the network

    NOTE: This represents the current state of the machine as observed by the agent, and not the
    TOTAL state of the machine.

    The host will keep track of the following properties:
    board_address [tuple] - e.g (0, 1) indicating subnet 0 and host 1. This will allow us to construct a matrix
                            of what machines can communicate with what other machines so the agent can learn to understand pivoting

    ip_address [str] - e.g 127.0.0.1 this will indicate the actual address of the machine that metasploit can use to run exploits

    services [dict] - will start with a dict of all possible services as keys. the values will then be metasploit dictionaries representing service info

    vulns [dict] - will start with list of all vuln as keys, the values will then be metasploit dictionaries representing the vulns

    loots int - the number of loots acquired.

    creds - a dictionary of the level of access for creds, set to true if we have credentials with that access. also a reference to a file containing the login creds
    """

    def __init__(
        self,
        board_address,
        ip_address,
        services,
        vulns=None,
        loot=None,
        creds=None,
        console=None,
        session=None,
        vector_size=None,
    ):
        self.board_address = board_address
        self.ip_address = ip_address
        self.services = self._construct_services(services)
        self.vector_size = vector_size
        self.loots = len(loot)
        self.has_creds = 1 if creds else 0
        self.has_open_console = 1 if console else 0  # denotes regular command shell
        self.has_open_session = 1 if session else 0  # denotes meterpreter
        self.vulns = vulns

    def _construct_services(self, services):
        """Use SERVICES_TCP in utils to create a dictionary of top services

        Args:
            services (dict): Dictionary representing current services in host

        Returns:
            dict: Dictionary representing active services in the host
        """
        services_tcp = SERVICES_TCP.copy()
        for service in services:
            if service["port"] in services_tcp:
                target_service = services_tcp[service["port"]]
                if service["name"] == target_service["service"]:
                    target_service["status"] = True
        return services_tcp

    @property
    def service_count(self):
        active_services = 0
        for port in self.services.keys():
            service = self.services[port]
            if service["status"]:
                active_services += 1
        return active_services

    @property
    def vuln_count(self):
        return len(self.vulns)

    @property
    def credentialed_access(self):
        return self.has_creds

    @property
    def loot_count(self):
        return self.loots

    @property
    def meterpreter_shell(self):
        return self.has_open_session

    @property
    def open_console(self):
        return self.has_open_console

    def vectorize(self):
        vector = np.zeros(self.vector_size, dtype=np.float32)
        return vector
