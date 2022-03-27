SERVICES_TCP = {
    80: {"service": "http", "status": False},
    23: {"service": "telnet", "status": False},
    443: {"service": "http", "status": False},
    21: {"service": "ftp", "status": False},
    22: {"service": "ssh", "status": False},
    25: {"service": "smpt", "status": False},
    3389: {"service": "ms-term-server", "status": False},
    110: {"service": "POP3", "status": False},
    445: {"service": "Microsoft-DS", "status": False},
    139: {"service": "NetBIOS-SSN", "status": False},
    5432: {"service": "postgresql", "status": False},
}

REWARDS = {
    "vulns": 10,
    "services": 5,
    "creds": 15,
    "loot": 1,
    "root": 20,
    "shell": 5,
    "meterpreter": 8,
}
