from setuptools import setup, find_packages

setup(
    name='df1',
    version='0.0.18',
    description='df1',
    url='https://github.com/djeethub/df1.git',
    packages=find_packages(
        where='src',
        include=['df1'],
    ),
    package_dir={"": "src"},
    install_requires=[
      'torch', 'torchvision',
      'diffusers[torch]', 'transformers', 'accelerate', 'scipy', 'safetensors', 'compel', 'k-diffusion', 'omegaconf',
#      'git+https://github.com/djeethub/upscaler.git', 'git+https://github.com/djeethub/df_helper.git'
   ],
)