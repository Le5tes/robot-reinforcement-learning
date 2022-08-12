# Ant Environment Setup Guide

This guide is to help setting up the environment to run the Ant example from Gym OpenAI. The current TD3Agent notebook is written to train based on this environment.
To run Ant, [Mujoco](https://www.gymlibrary.ml/environments/mujoco/) must be installed. The rest of this guide is for this purpose. 

The guide was tested on Ubuntu 18.04 LTS and 22.04 LTS. Windows system was not recommended due to limitation of Path Length issue encountered during installation, unless you are experienced in setting Environment Variables

Python version is 3.7.13. Python 3.10 is not recommended for some compatible issues encountered.

## Procedure
It is recommended to create a virtual environment, and do all actions in that environment. 
1. Download your system version of Mujoco 150 (**NOT 200**) engine from http://www.roboti.us/download.html, unzip, put folder **mjpro150** in
**/home/{youraccount}/.mujoco/**

2. Download license http://www.roboti.us/license.html, put the txt file in the same folder where mjpro150 is

3. Add path to .bashrc: export LD_LIBRARY_PATH=/home/{youraccount}/.mujoco/mjpro150/bin. Then source .bashrc. (other installing problem for mujoco py see https://github.com/openai/mujoco-py)
4. Upgrade pip to latest version by: pip install --upgrade pip
5. Install dependencies:
- pip install lockfile numpy matplotlib
6. pip install -U mujoco-py==1.50.1.68 
7. pip install gym[mujoco]
