"""Privilege Escalation actions
"""
from .action import PrivilegeEscalation


class GetSystem(PrivilegeEscalation):
    def __init__(self):
        """Metasploit literally just tries tons of stuff
        Including but not limited to this stuff https://cd6629.gitbook.io/ctfwriteups/windows-privesc
        aka services running as system with user-configurable startup binaries and stuff.

        https://www.offensive-security.com/metasploit-unleashed/privilege-escalation/

        """
        super(GetSystem, self).__init__()
        self.req_platform = "windows"


class GetRoot(PrivilegeEscalation):
    """wherein our 'action' is actually trying the linux suggester, parsing its contents to determine if any are viable, and running those that are

    aka the attached link shows documentation of a metasploit module that, when run, suggests a series of
    local privilege escalation techniques for linux, and whether it seems like the machine in question is vulnerable. we can then run any/all of these,
    basically the same as 'getsystem

    https://null-byte.wonderhowto.com/how-to/get-root-with-metasploits-local-exploit-suggester-0199463/
    """

    def __init__(self):
        super(GetRoot, self).__init__()
        self.req_platform = "linux"
