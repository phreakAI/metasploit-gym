""" Handles the initialization of the metasploit-gym through the command line interface.
"""

import argparse
import time

import metasploit_gym


def _get_args():
    """Parses the command line arguments and returns them."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        choices=["human", "random"],
        help="The execution mode for the game.",
    )

    return parser.parse_args()


def random_agent_env():
    env = metasploit_gym.make("metasploit_gym:MetasploitNetworkEnv-v0")
    env.reset()
    score = 0
    while True:
        env.render()

        # Getting random action:
        action = env.action_space.sample()

        # Processing:
        obs, reward, done, _ = env.step(action)

        score += reward
        print(f"Obs: {obs}\n" f"Action: {action}\n" f"Score: {score}\n")

        time.sleep(1 / 30)

        if done:
            env.render()
            time.sleep(0.5)
            break


def main():
    args = _get_args()

    if args.mode == "human":
        metasploit_gym.network.main()
    elif args.mode == "random":
        random_agent_env()
    else:
        print("Invalid mode!")
