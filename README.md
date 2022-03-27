# MetasploitGym

MetasploitGym is a [gym](https://github.com/openai/gym) environment designed to allow RL agents to interact with the[Metasploit Framework's](https://github.com/rapid7/metasploit-framework) gRPC service to interact with networks and singular machines. 

**Note**: MetasploitGym is research code and under active development. Breaking changes are not just likely, but necessary to get the API where it needs to be. In fact, this will all probably be completely rewritten.

`examples` features stubs for a DeepQNetwork, Random, and Keyboard driven agents. 

`tests` Need to be filled out. Currently mostly makes sure the gRPC service didn't break.


### Requirements

This software communicate with metasploit over g-RPC. To do this, you're going to need a g-RPC service running. Personally I've got a dockerfile to handle this. I've got a dockerfile set up how I like it [here](https://github.com/SJCaldwell/dockerfile-msf) based on [phocean's](https://github.com/phocean/dockerfile-msf) excellent work.

Once you start up the dockerfile, run the following commands

`./msfconsole`
`load msgrpc Pass=[your_pass] ServerPort=[your_port] ServerHost=0.0.0.0 SSL=true`

I've found when running the msfrpcd service I cannot maintain a database connection, I'll work to determine why that is later.

It's easier to test database changes directly through msfconsole directly though, so it's not the end of the world. 

Metasploit gym will also assume you have `METASPLOIT_PASSWORD` `METASPLOIT_PORT` and `METASPLOIT_HOST` as environmental variables it can use to connect to msgrpc. This will correspond to `[your_pass]`, `[your_port]`, and the hostname of your grpc service above. 

### API

MetasploitGym, while training, will undergo several episodes. It will do this by running its `env.reset()` function. In order to keep this general, it will require *the client* to write code that will reset its own environment. A VirtualBox sample for resetting a singular machine called `metasploitable` is included below.


```python
import pyvbox
import virtualbox

    def environment_reset():
        start = time.time()
        name = "gym_episode_start"
        vb = virtualbox.VirtualBox()
        session = virtualbox.Session()
        try:
            vm = vb.find_machine("metasploitable")
            snap = vm.find_snapshot(name)
            vm.create_session(session=session)
        except Exception as e:
            print(str(e))
            return False
        shutting_down = session.console.power_down()
        while shutting_down.operation_percent < 100:
            time.sleep(0.5)
        restoring = session.machine.restore_snapshot(snap)
        while restoring.operation_percent < 100:
            time.sleep(0.5)
        if restoring.completed == 1:
            print(f"Restore machine in {str(time.time() - start)} sec")
        vm = vb.find_machine("metasploitable")
        session = virtualbox.Session()
        vm.launch_vm_process(session, "gui", [])
```
However that code is written, as long as the function does not end until the environment is in a clean state, then you're ready to use the metasploit gym. 

```python
import metasploit_gym

env = metasploit_gym.metasploit_env.MetasploitNetworkEnv(
    reset_function=environment_reset,
    initial_target=target_host
)
```
It will default to having the state space for 1 subnet and 1 host. This can be changed with the `max_subnets`, `max_hosts_per_subnet` arguments. You can define an initial target as well, which will be where the agent starts. 

### Release Notes
MetasploitGym is under active development, and release notes will be stored in the [releases page](https://github.com/phreakai/metasploitgym/releases) on GitHub. 
