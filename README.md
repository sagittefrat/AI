# AI

## Reinforcemnt Learning
In this repository you can find different codes visuallizes how different learning algorithms and parameters can change the solution to the RL problem. 

### Algorithms:
    
   <img src="https://github.com/sagittefrat/AI/blob/master/QL.png" />

### Actions Choosing:
   Epsilon-Greedy
   
   Softmax (Boltzman Temperture) <img src=https://github.com/sagittefrat/AI/blob/master/softmax.png />	
   
   Random
### RL parameters:
	exploration rate (epsilon), exploration rate decay
	learning rate (alpha), learning rate decay
	discount factor (gamma)
	inverse temperature (beta), inverse temperature increase
	

### Enviornments:
#### FrozenLake https://gym.openai.com/envs/FrozenLake-v0
1 dimensional discrete state space with one dimensional discrete actions space: (F,S,H,G)

When at state *s* it is important to take the right action *a* in order to get to the goal faster/ achieve more reward.
* here you can see how choosing an action is changing the reward - random policy vs softmax policy:
<img src="https://github.com/sagittefrat/AI/blob/master/results/results-QL-FrozenLake-v0-random-q_learning/reward.png" width="420" title="Initial agent"/><img src="https://github.com/sagittefrat/AI/blob/master/results/results-QL-FrozenLake-v0-softmax-q_learning/reward.png" width="420" title="Final agent 300 episodes"/> 
as we expected, at the beggining explore more and towards the end exploit more is giving more reward than just exploring



#### MountainCar https://gym.openai.com/envs/MountainCar-v0
2 dimensional continous state space: (position, velocity) with one dimensional discrete actions space: (left, neutral, right)

The *learning rate* will determine how much importance we give to new knowledge, giving it less importance can result in slower convergence, giving it high importance can lead to forgetting more and in case of noisy observation can result in inaccurate prediction
* here you can see in Sarsa with Softmax at episode 125 how changing the Learning rate changes the system behaviour - learning rate-0.5 vs learning rate-0.1:

	<img src="https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-softmax-sarsa-lr0.5-video000125.gif" width="375" title="Initial agent"/> <img src="https://github.com/sagittefrat/AI/blob/master/results/GIFs/results-QL-MountainCar-v0-softmax-sarsa-lr0.1-video000125.gif" width="375" title="Final agent 300 episodes"/> 
	
The *number of episodes* will determine how much knowledge on the enviornment we recieve. As more exprience we expect better results (that can lead to overfitting) 	
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

<img src=https://github.com/sagittefrat/AI/blob/master/RL.png/>

