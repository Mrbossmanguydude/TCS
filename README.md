# TCS
Computer Science NEA project. A proof-of-concept simulation testing whether a single centralised neural network can coordinate autonomous vehicles more efficiently than decentralised control. Uses curriculum-based reinforcement learning with PPO to minimise congestion, collisions, and journey times across procedurally generated road networks. 

The goal is to make multiple agents, capable of percieving a state within the procedurally generated road network, work out a way to optimise their pathfinding and speed such that they minimise the factors above, as well as getting as close to the often unachievable optimum of simple individualistic pathfinding.

In this case, simple individualistic pathfinding is an even more abstract version of the intial simulation, such that each agent uses an optimal pathfinding algorithm, at max speed, while ignoring other agents in case of collisions. This is the theoretical limit to how good the reward can be, and is shown as a metric to the user to evaluate.

For now, the network functions like a typical PPO, the process usually goes:
  - Generate state, destinations and pre-training compute (if not already available).
  - Observe the current state, and use it as an input for the features (information is conveyed via a feature vector per vehicle (agent), containing features such as location, relative location to other agents, distance to goal, collision amt, current speed, proximity to road, current turning angle (or discrete direction in initial phases) and more to be added as the agent is trained, dependent on what information will best inform the agents about the environment to help it reach its goals).
  - The PN (Policy Network) computes an action for each vehicle as an output of a vector of the same size as the input vector (so per vehicle), to output changes in speed and turning (+ perhaps more in the future).
  - Based on f(s_t, a_t, s_t+1) (the reward function), the value of the reward is determined, dependent on the current state, the next state (after implemented trajectories) and the action for the vehicle to get there.
  - The trajectory (a_t, s_t, r_t, s_t+1) is logged, and stored as part of the data (for purposes of efficiency, it is likely that only a_t and r_t will be stored, as well as the initial state, from which we can derive the next states, this reduces the amount of space needed).
  - The return is calculated dependent on a preset (by user) hyperparam gamma, and the VN (Value Network) calculates the predicted return. (So loss simply becomes expectation at t, of MSE of the VN)
  - After this, the advantage is calculated via return - VN_pred.
  - The clipped surrogate objective finds the expectation of the minimum of the unclipped and clipped objective (dependent on epsilon, advanatge and the ratio of the old policy to the new policy)
  - Therfore we are able to compute combined and individual losses based on c1 and c2 (weighting coefficients dependent on stage of curriculum) as well as an entropy bonus to encourage exploration.
  - Backpropogation then occurs using these losses, after which the episode steps another unit in time.
  - This process is repeated for hundreds of episodes, until the first phase of training is complete (has met a good standard of loss, and improvement is noticed within the evaluation).
  - Phases continue in this way until the final one is complete, the network undergoes a final evaluation and it is decided whether it is good enough or not.

As my peripherals may not possess the computational power to find a good minimum of loss for both the VN and PN, a result that leads to a failure in understanding of the environment by the agent (it does not converge to a "good" solution) within the given time, we cannot definitively say whether or not that training (in this way) can actually succeed or not to make a system that is capable of having "intelligent" agents that can navigate the state space with an optimal policy. 
Hence, given the broader aim of this project, which is to assess whether a system like this of self-driving vehicles that confer with a centralised controller in order to move around better than humans, the result of the project will more than likely be an indication of how far this paticular architecture can take the avenue, and if more improvements are needed or not or if another type of solution would be either simpler, cheaper, more optimal or a combination of these elements.
