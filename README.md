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

* Sarsa with Softmax episode 125 learning rate-0.5 vs learning rate-0.1:

	<img src="https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-softmax-sarsa-lr0.5-video000125.gif" width="375" title="Initial agent"/> <img src="https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-softmax-sarsa-lr0.1-video000125.gif" width="375" title="Final agent 300 episodes"/> 
	
	
* Epsilon-greedy with Q-learn episode 8 vs Q-learn episode 64 vs Sarsa episode 64 (learning rate-0.1):

	<img src=https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-epsilon_greedy-q_learning-lr0.1-video000008.gif width="250" title="Initial agent"/>
	<img src=https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-epsilon_greedy-q_learning-lr0.1-video000064.gif width="250" title="Final agent 100 episodes"/>
	<img src=https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-epsilon_greedy-sarsa-lr0.1-video000064.gif width="250" title="Final agent 100 episodes"/>


### How to run:
*Use SARSA/Q-learning algorithm with epsilon-greedy/softmax policy*
* *FrozenLake env*: FrozenLake_QL.py, FrozenLake_QL_options.py 
* *MountainCar env*: MountainCar_QL.py, MountainCar_QL_options.py   

*Use **Deep** SARSA/Q-learning algorithm with epsilon-greedy/softmax policy*
* *FrozenLake env*: FrozenLake_DQN.py, FrozenLake_DQN_options.py 
* *MountainCar env*: MountainCar_DQN.py, MountainCar_DQN_options.py 

you can play with the different parameters, as shown in each options file, for help run:

	python MountainCar_QL.py -h
	
run with default parameters:

	python MountainCar_QL.py
	
run Algorithm - sarsa, Policy - softmax Learning Rate - 0.2 Number of episodes - 300 :

	python MountainCar_QL.py -a sarsa -p softmax -lr 0.2 -n 300



