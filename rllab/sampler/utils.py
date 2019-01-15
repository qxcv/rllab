import inspect
import numpy as np
from rllab.misc import tensor_utils
import time


def get_inner_env(env):
    if hasattr(env, 'wrapped_env') and env.wrapped_env is not env:
        return get_inner_env(env.wrapped_env)
    elif hasattr(env, 'env') and env.env is not env:
        return get_inner_env(env.env)
    return env


def has_det_kwarg(function):
    """Check that an action-taking method has a 'det' kwarg. This is the
    duck-type-y way of figuring out whether we can do deterministic sampling
    :)"""
    aspec = inspect.getfullargspec(function)
    num_kwargs = len(aspec.defaults)
    return len(aspec.args) > 2 \
        and num_kwargs > 0 \
        and 'det' in aspec.args[-num_kwargs:]


def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            animated_save_path=None,
            # set this to True(ish) to do deterministic action selection
            det=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()

    if det is not None:
        assert has_det_kwarg(agent.get_action), \
            "%r.get_action does not support det= kwarg; can't do " \
            "deterministic sampling" % (agent, )

    def get_action(o):
        if det is not None:
            return agent.get_action(o, det=det)
        return agent.get_action(o)

    # I don't think Gym envs have these attributes, but RLLab envs do
    has_flatten_o = hasattr(env.observation_space, 'flatten')
    has_flatten_a = hasattr(env.action_space, 'flatten')

    path_length = 0
    if animated:
        if animated_save_path is None:
            env.render()
        else:
            from skvideo.io import FFmpegWriter
            inner_env = get_inner_env(env)
            vid_writer = FFmpegWriter(animated_save_path)
            frame = inner_env.render(mode='rgb_array')
            vid_writer.writeFrame(frame)
    try:
        while path_length < max_path_length:
            a, agent_info = get_action(o)
            next_o, r, d, env_info = env.step(a)
            flat_o = o
            if has_flatten_o:
                flat_o = env.observation_space.flatten(o)
            observations.append(flat_o)
            rewards.append(r)
            flat_a = a
            if has_flatten_a:
                flat_a = env.action_space.flatten(a)
            actions.append(flat_a)
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
