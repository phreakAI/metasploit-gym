from gym import Env, spaces
from gym.utils import seeding
from pymetasploit3.msfrpc import MsfRpcClient
from .action.action import Action, Exploit, Scan
from .action.exploit import *
from .action.scan import *
from .host.network import Host, Network
from dotenv import load_dotenv
import os
import time


class MetasploitEnv(Env):
    def __init__(self):
        """
        Will have all the required gym functionality and then also necessary stuff for our other two environments
        """
        self.action_space = None  # Space object corresponding to valid actions
        # Space object corresponding to valid observations
        self.observation_space = None
        self.reward_range = None  # A tuple corresponding to the min and max possible rewards (default [-inf, +inf] )

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()` to reset this
        environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): An action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float): amount of reward returned after previous acgtion
            done (bool): whether the episode has ended, in which case further step() calls return undefined results
            info (dict): Returns auxiliary diagnostic info
        """
        raise NotImplementedError

    def reset(self):
        """Resets the environment to an initial state and returns an initial observation.

        Note that this function should not reset the environment's random number generators;
        random variables in the environment's state should be sampled independently between multiple calls to `reset()`. In
        other words each call of `reset()` should yield an environment suitable for a new episode, independent
        of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. By convention, if mode is:

        - human: render to the current display or terminal and return nothing for human consomptuon.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3), representing RGB values for x-by-y pixel image, suitable
        for turning into a video.
        - ansi: Return a string (str) or StringeIO.StringIO containing a terminal-style text representation

        Note:
            Make sure that your class's metadata `render.modes` key includes
            the list of supported mode. It's recommended to call super() in implementations
            to use the functionality of this method

        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception

        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform necessary cleanup.

        Environments will automatically close() themselves when garbage collected or on program exit.
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator.

        Note:
            Some environments use multiple psuedorandom number generators.
            We want to capture all such seeds used in order to ensure that there aren't
            accidental correlations between multiple generators.

        Returns:
        list<bigint>: Returns the list of seeds used in this env's random number generators. The first
        value in the list should be the "main" seed, or the value which a reproducer should pass to 'seed'. Often,
        the main seed equals the provided 'seed', but this won't be true if seed=None, for example.

        Args:
            seed ([type], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError


class MetasploitNetworkEnv(MetasploitEnv):
    def __init__(self, reset_function, initial_host="127.0.0.1", max_subnets=1, max_hosts_per_subnet=1, total_hosts=1):
        super().__init__()
        load_dotenv()
        self.environment_reset_function = reset_function
        self.client = self.create_client()
        self.client.db.workspaces.add("metasploitgym")
        self.client.db.workspaces.set("metasploitgym")
        self.total_hosts = total_hosts
        # TODO: Replace this with CIDR address later
        self.target_host = initial_host
        if self.target_host is None:
            raise ValueError("Set TARGET_HOST in .env to use the metasploit network env")
        self.host_dict = {self.target_host: (0, 0)}  # map vector to IP
        self.tcp_services = dict()  # list of all services, and a 1 if it's up
        self.udp_services = dict()
        self.privileges = dict()
        self.loot = dict()
        self.max_subnets = max_subnets
        self.max_hosts_per_subnet = max_hosts_per_subnet
        self.action_space = FlatActionSpace(max_subnets=max_subnets, max_hosts_per_subnet=max_hosts_per_subnet)
        self.network = Network(
            max_subnets=max_subnets,
            max_hosts_per_subnet=max_hosts_per_subnet,
            num_exploits=len(self.action_space),
        )
        self.observation_space = spaces.Box(low=0, high=self.network.max_reward(), shape=self.network.vectorize().shape)

    def create_client(self):
        metasploit_pass = os.getenv("METASPLOIT_PASSWORD", default=None)
        metasploit_port = os.getenv("METASPLOIT_PORT", default=None)
        metasploit_host = os.getenv("METASPLOIT_HOST", default=None)

        if metasploit_host is None or metasploit_port is None or metasploit_pass is None:
            raise ValueError(
                "Please include a .env file with METASPLOIT_PASS, METASPLOIT_PORT, and METASPLOIT HOST set to the values of the msgrpc service"
            )
        client = MsfRpcClient(metasploit_pass, server=metasploit_host, port=metasploit_port, ssl=True)
        return client

    def calculate_board_address(self, ip):
        if ip in self.host_dict:
            return self.host_dict[ip]
        proposed_subnet = ip.split(".")[-2]
        amount_in_subnet = 0
        subnet_idx = None
        highest_subnet = 0
        for ip in self.host_dict.keys():
            if (self.host_dict[ip])[0] > highest_subnet:
                highest_subnet = self.host_dict[ip][0]
            if ip.split(".")[-2] == proposed_subnet:
                amount_in_subnet += 1
                subnet_idx = self.host_dict[ip][0]  # first part of tuple
        if amount_in_subnet > self.max_hosts_per_subnet:
            return None
        elif subnet_idx == None and highest_subnet >= self.max_hosts_per_subnet - 1:
            return None
        else:
            if subnet_idx is None:
                subnet_idx = highest_subnet + 1
                return (subnet_idx, 0)
            else:
                return (subnet_idx, amount_in_subnet)

        ### see if we have any slots left in this subnet

    def update_env(self):
        # get hosts. pull info for hosts in order
        # first check if we have open sessions and organize them by host
        # return new reward
        reward = 0
        open_meterpreter_sessions = {}
        open_console_sessions = {}

        session_keys = self.client.sessions.list
        for sid in session_keys:
            if session_keys[sid]["type"] == "meterpreter":
                hostname = session_keys[sid]["target_host"]
                open_meterpreter_sessions[hostname] = session_keys[sid]
            if session_keys[sid]["type"] == "shell":
                hostname = session_keys[sid]["target_host"]
                open_console_sessions[hostname] = session_keys[sid]

        hosts = self.client.db.workspaces.current.hosts.list  # list of hosts we know about
        for host in hosts:
            address = host["address"]
            services = self.client.db.workspaces.current.services.find(
                addresses=[address]
            )  # list of services for those hosts
            vulns = self.client.db.workspaces.current.vulns.find(addresses=[address])  # list of exploited vulns so far
            loot = self.client.db.workspaces.current.loots.find(addresses=[address])  # loot taken from machines
            creds = self.client.db.workspaces.current.creds.find(addresses=[address])  # credentials discovered
            if address in open_meterpreter_sessions:
                has_session = 1
            else:
                has_session = 0
            if address in open_console_sessions:
                has_console = 1
            else:
                has_console = 0
            # CALCULATE BOARD ADDRESS
            board_address = self.calculate_board_address(ip=address)
            if board_address:
                potential_host = Host(
                    board_address=(0, 1),
                    ip_address=address,
                    services=services,
                    vulns=vulns,
                    loot=loot,
                    creds=creds,
                    console=has_console,
                    session=has_session,
                    vector_size=self.network.host_vector_size,
                )
            else:
                continue
            if self.network.check_host_exists(potential_host.board_address) is True:
                reward += self.network.compare_updated_host(potential_host)
                self.network.update_host(potential_host)
            else:
                reward += self.network.add_host(potential_host)
            # need to find way to derive privileges
        return reward

    def step(self, action):
        """
        This function validates the action that is to be taken, making sure it is a legal action.
        That includes that the subnet and host exist in the lookup table, and that the action can correctly be applied to the host.
        :param action: an object of type Action
        :return: Observation, reward, done, debug info
        """
        if not issubclass(type(action), Action):
            raise TypeError("Only actions of type Action can be processed by the MetasploitNetworkEnv")
        board_addr_to_host = dict([(value, key) for key, value in self.host_dict.items()])

        if action.target in board_addr_to_host:
            host_addr = board_addr_to_host[action.target]
        else:
            raise KeyError(f"Chosen target {action.target} does not exist in current host_dict")
        print(action)
        action.execute(self.client, host_addr)
        time.sleep(5)  # let end of execution play out, for example VSFTPD
        # use kwargs to just pass all the info we have and let the action decide what it needs?
        # load relevant service port
        # - the host the action is being taken against
        reward = self.update_env()  # update env
        # calculate reward based on
        debugging_info = {}
        return self.network.vectorize(), reward, self.goal_reached(), debugging_info

    def reset(self):
        # reset data structures that make up network
        # return initial observation
        self.host_dict = {self.target_host: (0, 0)}
        self.tcp_services = dict()
        self.udp_services = dict()
        self.privileges = dict()
        self.loot = dict()
        self.network = Network(
            max_subnets=self.max_subnets,
            max_hosts_per_subnet=self.max_hosts_per_subnet,
            num_exploits=len(self.action_space),
        )
        # have to remove sessions seperately, lets close them
        for session in self.client.sessions.list:
            session_obj = self.client.sessions.session(session)
            session_obj.stop()
        # reset client database by removing current workspace, then add new workspace with same name
        self.client.db.workspaces.remove("metasploitgym")
        # then add a new one
        self.client.db.workspaces.add("metasploitgym")
        self.client.db.workspaces.set("metasploitgym")
        self.environment_reset_function()
        return self.network.vectorize()

    def goal_reached(self):
        # check to see if all hosts have network access
        if self.network.hosts == []:
            return False
        for host in self.network.hosts:
            if host.meterpreter_shell == 0 and host.open_console == 0:
                return False
        return True


class MetasploitSimulatorEnv(MetasploitEnv):
    # TODO: Build Simulator Environment as POMDP
    def __init__(self):
        raise NotImplementedError


class FlatActionSpace(spaces.Discrete):
    """Flat Action space"""

    def __init__(self, max_subnets, max_hosts_per_subnet):
        self.max_subnets = max_subnets
        self.max_hosts_per_subnet = max_hosts_per_subnet
        self.actions = self.generate_action_list()
        super().__init__(len(self.actions))

    def generate_action_list(self):
        action_list = []
        for i in range(self.max_subnets):
            for j in range(self.max_hosts_per_subnet):
                host_idx = (i, j)
                for ScanAction in Scan.__subclasses__():
                    scan = ScanAction((i, j))
                    action_list.append(scan)
                for ExploitAction in Exploit.__subclasses__():
                    action = ExploitAction((i, j))
                    action_list.append(action)
        return action_list
        # need to generate each possible action for each possible host
        # requires network description

    def get_action(self, action_idx):
        """Action has to be an index"""
        assert isinstance(action_idx, int), "When using a flat action space must be an integer"
        assert action_idx <= len(self.actions) - 1, "Action can't be longer than list"
        return self.actions[action_idx]

    def __len__(self):
        return len(self.actions)
