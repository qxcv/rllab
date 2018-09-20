import numpy as np
from rllab.misc import tensor_utils
import time


def get_inner_env(env):
    if hasattr(env, 'wrapped_env') and env.wrapped_env is not env:
        return get_inner_env(env.wrapped_env)
    elif hasattr(env, 'env') and env.env is not env:
        return get_inner_env(env.env)
    return env


def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            animated_save_path=None):
    print('doing a thing')
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    print('resetting the environment')
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        print('rendering thing')
        if animated_save_path is None:
            print('DOING IT THE DUMB WAY')
            env.render()
        else:
            print('doing it the smart way')
            from skvideo.io import FFmpegWriter
            inner_env = get_inner_env(env)
            vid_writer = FFmpegWriter(animated_save_path)
            frame = inner_env.render(mode='rgb_array')
            vid_writer.writeFrame(frame)
    try:
        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            observations.append(env.observation_space.flatten(o))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                if animated_save_path is None:
                    env.render()
                    timestep = 0.05
                    time.sleep(timestep / speedup)
                else:
                    frame = inner_env.render(mode='rgb_array')
                    vid_writer.writeFrame(frame)
    finally:
        if animated and animated_save_path is not None:
            vid_writer.close()
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos), )
