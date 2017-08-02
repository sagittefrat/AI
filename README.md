# AI

## Reinforcemnt Learning

### Algorithms:
	Q-Learning  	Off-policy
	Sarsa		On-policy
### Choosing Actions:
	Epsilon-Greedy
	Softmax (Boltzman Temp)
	Greedy
	Random

### Enviornment
#### FrozenLake https://gym.openai.com/envs/FrozenLake-v0
has a 1 dimensional discrete state space with one dimensional discrete actions space: (F,S,H,G)


* Sarsa with Softmax:
	<img src="https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000008.gif" width="375" title="Initial agent"/> <img src="https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000216.gif" width="375" title="Final agent 300 episodes"/> 

	![animation](https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000008.gif)  
	|
	![animation](https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-sarsa-softmax/openaigym.video.0.30505.video000216.gif)
	
	
* Q-learn with epsilon greedy:
<figure>
  <img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000000.gif width="375" title="Initial agent"/>
  <figcaption>Initial agent</figcaption>
  <img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000000.gif width="375" title="Initial agent"/>
  <figcaption>Initial agent</figcaption>
</figure>
	<"Initial agent" width="375" />  <"Final agent 100 episodest" width="375" /> 
	<img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000000.gif width="375" title="Initial agent"/>
	<img src=https://github.com/sagittefrat/AI/blob/master/results-%3CTimeLimit%3CMountainCarEnv%3CMountainCar-v0%3E%3E%3E-q_learning-epsilon_greedy/openaigym.video.0.24474.video000064.gif width="375" title="Initial agent"/>
	


#### MountainCar https://gym.openai.com/envs/MountainCar-v0
has a 2 dimensional continues state space: (position, velocity) with one dimensional discrete actions space: (left, neutral, right)




