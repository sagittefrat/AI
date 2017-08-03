# AI

## Reinforcemnt Learning

### Algorithms:
	Q-Learning  	Off-policy
	Sarsa		On-policy
### Actions Choosing:
	Epsilon-Greedy
	Softmax (Boltzman Temp)
	Greedy
	Random

### Enviornments:
#### FrozenLake https://gym.openai.com/envs/FrozenLake-v0
has a 1 dimensional discrete state space with one dimensional discrete actions space: (F,S,H,G)
	


#### MountainCar https://gym.openai.com/envs/MountainCar-v0
has a 2 dimensional continues state space: (position, velocity) with one dimensional discrete actions space: (left, neutral, right)

* Sarsa with Softmax episode 8 vs episode 216:

	<img src="https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000008.gif" width="375" title="Initial agent"/> <img src="https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000216.gif" width="375" title="Final agent 300 episodes"/> 
	
	
* Q-learn with epsilon greedy episode 8 vs episode 64:

	<img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000008.gif width="375" title="Initial agent"/>
	<img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000064.gif width="375" title="Final agent 100 episodes"/>

### How to run:
*Use SARSA/Q-learning algorithm with epsilon-greedy/softmax policy*
* **FrozenLake env**: FrozenLake_QL.py, FrozenLake_QL_options.py 
* **MountainCar env**: MountainCar_QL.py, MountainCar_QL_options.py   

*Use **Deep** SARSA/Q-learning algorithm with epsilon-greedy/softmax policy*
* **FrozenLake env**: FrozenLake_DQN.py, FrozenLake_DQN_options.py 
* **MountainCar env**: MountainCar_DQN.py, MountainCar_DQN_options.py 

you can play with the different parameters, as shown in each options file, for help run:

	python MountainCar_QL.py -h
	
run with default parameters:

	python MountainCar_QL.py
	
run Algorithm - sarsa, Policy - softmax Learning Rate - 0.2 Number of episodes - 300 :

	python MountainCar_QL.py -a sarsa - p softmax -lr 0.2 -n 300



