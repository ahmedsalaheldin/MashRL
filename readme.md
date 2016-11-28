This repository presents a framework to train intelligent agents to perform a navigation task using deep RL learning. Deep supervised learning is used to shape a reward from demonstrations. 
This repository heavily relies on this implementation of DQN https://github.com/spragunr/deep_q_rl.

The agent acts in a 2D simulated environment and learns from raw visual information. The environment consists of a grid where each cell represents a state. A state is represented by an image of the cell number. The target is to reach a specific cell on the grid. The Grid can be set to any size and the environment is generated automatically.

##usage

This repository consists of a number of modules.

-The 2D simulator acts as a server that builds an environment for a selected task. Client agents can then act in that environment by sending an action to the server. The simulator calculates changes to the world based on the received action and sends the current status (in the form of an image) back to the client.

-A stand alone deep learning module learns a reward shaping function that creates a mapping between observation( an image) and a potential reward from a collected set of demonstrations. The learned model is saved and can be used by the reinforcement learning algorithm to form the target rewards. 

-The prediction server receives an image from the client and learns a policy using deep reinforcement learning with reward shaping. The server returns a predicted action based on the trained policy to the client.

-The agent Client which interacts with the simulation and prediction servers.

A typical workflow of the system looks like this:
	-Data collection
		-Run simulation server in Play mode. 
		-A deterministic policy performs the optimal actions to solve the task.
		-These demonstrations are saved as a data set of observation, action pairs.
		-After a sufficient amount of training samples is collected this process is terminated
	-Training a deep reward shaping network
		-The supervised learning script uses the saved dataset to train a neural network to shape rewards from observations.
		-The trained model is saved.
	-Learning a policy using deep reinforcement learning
		-A client connects to the simulation server
		-The client also connects to a prediction server which loads the saved reward shaping network.
		-The client receives a frame from the simulator and sends it to the prediction server.
		-The prediction replies with an action generated off-policy
		-The client receives an action from the prediction server and sends it to the simulator.
		-The action is performed in the simulator and a reward is returned to the client which sends it to the prediction server.
		-The rewards along with the corrosponding states and actions are used to train the reinforcement learning algorithm
		-This process is repeated until the desired number of training epochs is reached.
		- During testing, the learned policy is used to generate actions (instead of off-policy random trials) and send them to the client.

The reinforcement learning algorithm can be set to use or ignore the shaped rewards when creating target rewards for the cost function.

## Implementation details

TrainRewardShaping.py Script for training the reward shaping neural network from a dataset of demonstrations

imageMatrixTask.py Creates the simulated environment and starts the server.

/deep_q_rl/deep_q_rl/run_nature.py The prediction server. Takes reward shaping network file as argument.

RLAgent.cpp The client used to interact with the simulation and prediction servers


## Dependencies

Following is a list of the required dependencies:

ZeroMQ: A library for interprocess communication. Used to to send and receive messages between the agent and the prediction server. Python and C++ bindings are needed
http://zeromq.org/intro:get-the-software

Theano: The deep learning library used to train the agent
http://deeplearning.net/software/theano/install.html

Lasagne: A Library used to build and train neural networks in Theano.
https://github.com/Lasagne/Lasagne.git

rapidjson: A library for serializing messages to be send between processes
https://pypi.python.org/pypi/python-rapidjson/

Submodules:

ZeroMQ C++ headerfile
https://github.com/zeromq/cppzmq/blob/master/zmq.hpp

## Building
To build the client:
MashRL$ mkdir build
MashRL$ cd build 
MashRL/build$ cmake .. 
MashRL/build$ make

## Running
To start the simulator server
MashRL$ python imageMatrixTask.py

To start the prediction server
MashRL/deep_q_rl/deep_q_rl$ python run_nature.py

To start the prediction server with a reward shaping network
MashRL/deep_q_rl/deep_q_rl$ python run_nature.py --nn-file network_file.pkl

To run the supervised learning script for the reward shaping network
MashRL$ python TrainRewardShaping.py

To run the Client
MashRL/build$ ./RLAgent


