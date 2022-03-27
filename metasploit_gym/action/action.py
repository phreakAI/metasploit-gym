"""Action related classes for the MetasploitGym environment.

This module contains the different action classes that are used
to represent a subset of the abilities available to the metasploit API, along 
with the different ActionSpace and ActionResult classes.  
"""

class Action(object):
    """The base abstraction class for the environment.

    There are multiple types of actions. We consider exploits, scans, privilege escalations.
    """

    def __init__(self, name, target, req_access, req_os, req_version, **kwargs):
        """
        Args:
            name (str): name of action
            target (int, int): space of target (subnet, host)
            req_access (AccessLevel), optional: required access level to perform the action
            req_os (OperatingSystem), optional: required OS to perform the action
            req_version (RequiredVersion), optional: required version number
        """
        self.name = name
        self.target = target
        self.req_access = req_access
        self.req_os = (req_os,)
        self.req_version = req_version

    def is_exploit(self):
        """Check if action is exploit"""
        return isinstance(self, Exploit)

    def is_scan(self):
        """Check if action is scan"""
        return isinstance(self, Scan)

    def is_privilege_escalation(self):
        """Check if action is privilege escalation"""
        return isinstance(self, PrivilegeEscalation)

    def is_no_op(self):
        """Check if operation is none operation"""
        return isinstance(self, NoOp)

    def is_remote(self):
        """Check if action is remote. An action
        is remote if it's not being run from a process on the local
        machine"""
        return isinstance(self, (Scan, Exploit))

    def execute(self):
        """
        Execute the action
        """
        return NotImplementedError


class Exploit(Action):
    def __init__(self):
        super().__init__(
            self.name, self.target, self.req_access, self.req_os, self.req_version
        )


class Scan(Action):
    def __init__(self):
        raise NotImplementedError


class PrivilegeEscalation(Action):
    def __init__(self):
        raise NotImplementedError


class NoOp(Action):
    def __init__(self):
        raise NotImplementedError
