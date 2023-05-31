from setuptools import setup, find_packages

setup(
    name='df1',
    version='0.0.10',
    description='df1',
    url='https://github.com/djeethub/df1.git',
    packages=['df1'],
    install_requires=[
      'torch', 'torchvision',
      'diffusers[torch]', 'transformers', 'accelerate', 'scipy', 'safetensors', 'compel', 'k-diffusion', 'omegaconf',
#      'git+https://github.com/djeethub/upscaler.git', 'git+https://github.com/djeethub/df_helper.git'
   ],
)