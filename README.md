# REINFORCEMENT LEARNING ASSIGNMENT 2

Hi all, I've added the two notebook files to the directory 'src'.
We can put all our code files in there.

We could make another folder for the report/presentation, but also keep them in this repository so we can all make changes and collaborate.

## Getting started
If not already installed, download git from https://git-scm.com/downloads and go through the installer.

If not done so already, you will need to set up an ssh key, which github will use to authenticate you. See here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

Open up a terminal (git bash on windows), navigate to a folder where you want to store the files and run the command "git clone git@github.com:Le5tes/robot-reinforcement-learning.git"

This should create a directory called robot-reinforcment-learning, with this readme and our code files inside.

## Links
Gym jiminy code: https://github.com/duburcqa/jiminy

Gym jiminy documentation: https://duburcqa.github.io/jiminy/
- not great documentation

OpenAI Gym documentation: https://www.gymlibrary.ml/
- gym jiminy implements this API - the methods documented here are the same methods as GJ uses - so this is very useful

Keras documentation: https://keras.io/api/
- we will need to implement some neural nets so lets use keras?

## Useful commands
### Shell commands
Here are some useful commands for navigating when using a bash terminal (e.g. git bash)

- pwd
  - Shows which directory you are in
- cd <directory name>
  - moves to the directory
- ls
  - Lists the files in the directory you are currently in 

### Git commands
These are the basic git commands that will be most useful.
There are many more!

- git pull
  - Retrieves the latest code from github

- git add .
  - stages your changes for commiting

- git commit -m "some message"
  - makes a commit - a record of the state of the code with a message

- git push
  - pushes your commits to github