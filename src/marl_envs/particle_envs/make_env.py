"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""


def make_env(
    scenario_name,
    benchmark=False,
    max_epi_steps=25,
    discrete_action_space=True,
    discrete_action_input=False,
    config_name="antipodal",
    prey_accel=4.0,
    prey_max_v=1.3,
    obs_r=1.0,
    obs_resolution=8,
    flick_p=0.0,
    enable_boundary=False,
    discrete_mul=1,
    *args,
    **kargs
):
    """
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
        max_epi_steps   :   maximum time steps of one episode
        discrete_action_space  :    whether discrete action space or not
        discrete_action_input  :    see ./multiagent/environment.py
        config_name     :   the configuration file's name for advanced particle envs 
        prey_accel      :   prey's acceleration
        prey_max_v      :   prey's maximal speed
        obs_r           :   each agent's observation range
        obs_resolution  :   round agent's observation info 
        flick_p         :   a probability of flicking observation
        enable_boundary :   enable env's boundary or not
        discrete_mul    :   1 means discretize the action space into 4 directions; 
                            2 means discretize the action space into 8 directions;

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    """
    from .multiagent.environment import MultiAgentEnv
    import marl_envs.particle_envs.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    if scenario_name.startswith("simple"):
        if scenario_name.startswith("simple_coop_tag"):
            world = scenario.make_world(
                prey_accel=prey_accel,
                prey_max_v=prey_max_v,
                obs_resolution=obs_resolution,
                enable_boundary=enable_boundary,
            )
        else:
            world = scenario.make_world(
                obs_resolution=obs_resolution, enable_boundary=enable_boundary
            )

    if scenario_name.startswith("pomdp"):
        if scenario_name.startswith("pomdp_simple_coop_tag_v"):
            world = scenario.make_world(
                prey_accel=prey_accel,
                prey_max_v=prey_max_v,
                obs_r=obs_r,
                obs_resolution=obs_resolution,
                flick_p=flick_p,
                enable_boundary=enable_boundary,
            )
        if scenario_name == ("pomdp_advanced_spread"):
            world = scenario.make_world(scenarios.get_config(config_name), obs_r=obs_r)
        else:
            world = scenario.make_world(
                obs_r=obs_r,
                obs_resolution=obs_resolution,
                flick_p=flick_p,
                enable_boundary=enable_boundary,
            )

    if scenario_name == ("advanced_spread"):
        world = scenario.make_world(scenarios.get_config(config_name))

    # create multiagent environment
    if hasattr(scenario, "post_step"):
        post_step = scenario.post_step
    else:
        post_step = None

    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            scenario.state,
            scenario.env_info,
            discrete_action_space=discrete_action_space,
            discrete_action_input=discrete_action_input,
            max_epi_steps=max_epi_steps,
            discrete_mul=discrete_mul,
        )
    else:
        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            state_callback=scenario.state,
            env_info_call_back=scenario.env_info,
            post_step_callback=post_step,
            discrete_action_space=discrete_action_space,
            discrete_action_input=discrete_action_input,
            max_epi_steps=max_epi_steps,
            discrete_mul=discrete_mul,
        )
    return env
