"""An agent for interacting with the MetasploitGym environment using the keyboard.

To see available arguments, run python keyboard_agent --help
"""
import numpy as np
import metasploit_gym

LINE_BREAK = "-" * 60
LINE_BREAK_2 = "=" * 60


def choose_host(env):
    host_list = [host for host in env.host_dict.keys()]
    while True:
        try:
            print("KNOWN HOSTS:")
            for i in range(len(host_list)):
                print(f"[{str(i)}] IP: {host_list[i]}")
            idx = int(input("Please enter a number:"))
            return host_list[idx]
        except Exception:
            print("Invalid choice. Please select one of the numbered options")


def choose_action_for_host(env, ip_address):
    target_actions = []
    target = env.host_dict[ip_address]
    for action in env.action_space.actions:
        if action.target == target:
            target_actions.append(action)
    while True:
        try:
            print("ACTIONS ON HOST")
            for i in range(len(target_actions)):
                print(f"[{str(i)}] : action {target_actions[i].name}")
            idx = int(input("Please enter a number:"))
            return target_actions[idx]
        except Exception:
            print("Invalid choice. Please select one of the numbered options")


def run_keyboard_agent(env, step_limit=2, verbose=True):
    print(LINE_BREAK)
    print("STARTING EPISODE:")

    env.reset()
    total_reward = 0
    done = False
    t = 0

    while not done and t < step_limit:
        ip_address = choose_host(env)
        a = choose_action_for_host(env, ip_address)
        _, r, done, _ = env.step(a)
        print(r)
        total_reward += r
        if (t + 1) % 20 == 0 and verbose:
            print(f"t: {t}: reward: {total_reward}")
        t += 1

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE COMPLETE")
        print(LINE_BREAK)
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")
    return t, total_reward, done


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of random runs to perform (default=1",
    )
    args = parser.parse_args()
    run_steps = []
    run_rewards = []
    run_goals = 0

    def environment_reset():
        raise NotImplemented("Reset your network with this function")
        
    env = metasploit_gym.metasploit_env.MetasploitNetworkEnv(
        reset_function=environment_reset
    )
    for i in range(args.runs):
        steps, reward, done = run_keyboard_agent(env, step_limit=2, verbose=True)
        run_steps.append(steps)
        run_rewards.append(reward)
        run_steps = np.array(run_steps)
        run_rewards = np.array(run_rewards)

    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} " f"+/- {run_rewards.std():.2f}")
