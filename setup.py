# setup.py
from setuptools import find_packages, setup

setup(
    name='rllab',
    version='0.1.0',
    author='OpenAI',
    description='A framework for developing and evaluating reinforcement '
    'learning algorithms, fully compatible with OpenAI Gym',
    url='https://github.com/openai/rllab',
    packages=find_packages(
        '.',
        exclude=('contrib/', 'examples/', 'scripts/', 'tests/')),
    dependency_links=[
        # note that pip install requires --process-dependency-links!
        'git+https://github.com/Lasagne/Lasagne.git'
    ],
    install_requires=[
        # TODO: figure out which of these are truly necessary. Also split them
        # into install/dev dependencies.
        'numpy>=1.11',
        'scipy',
        'path.py',
        'python-dateutil',
        'joblib==0.9.4',
        'h5py',
        'matplotlib',
        'scikit-learn',
        'Pillow',
        'nose2',
        'tqdm',
        'pyzmq',
        'msgpack-python',
        'cached_property',
        'line_profiler',
        'cloudpickle',
        'Theano>=0.9.0',
        'lasagne>=0.2.dev1',
        'tensorflow>=1.0.1',
        # 'mako',
        # 'flask',
        # 'pygame',
        # 'boto3',
        # 'mujoco_py',
        # 'pybox2d',
        # 'PyOpenGL',
        # 'Cython',
        # 'git+https://github.com/plotly/plotly.py.git@2594076e29584ede2d09f2aa40a8a195b3f3fc66#egg=plotly',
        # 'git+https://github.com/openai/gym.git@385a85fd0c1b26ab1374f208fbb370e22647548d#egg=gym',
        # 'git+https://github.com/neocxi/prettytensor.git',
    ],
    zip_safe=False)
