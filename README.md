# **TCS – Computer Science NEA Project**

A proof-of-concept simulation testing whether a **single centralised neural network** can coordinate autonomous vehicles more efficiently than decentralised control.  
Uses **curriculum-based reinforcement learning (PPO)** to minimise **congestion, collisions, and journey times** across **procedurally generated road networks**.

---

The goal is to make multiple agents—each capable of perceiving a state within the procedurally generated road network—work out a way to optimise their pathfinding and speed such that they minimise the factors above, while still getting as close as possible to the often unachievable optimum of simple individualistic pathfinding.

In this case, **simple individualistic pathfinding** is an even more abstract version of the initial simulation: each agent uses an optimal pathfinding algorithm at maximum speed, ignoring other agents and possible collisions. This provides the **theoretical upper bound** of how good the reward can be, and is shown as a metric to the user for evaluation purposes.

---

## **Training Approach (High-Level)**

For now, the network functions like a typical PPO system. The process roughly goes:

1. **Generate state, destinations, and pre-training compute** (if not already available).
2. **Observe the current state**, and convert it into a **feature vector per vehicle**.  
   These features include:
   - location  
   - relative location to other agents  
   - distance to goal  
   - collision amount  
   - current speed  
   - proximity to road  
   - current turning angle (or discrete direction initially)  
   - and more features added gradually depending on training phase
3. The **Policy Network (PN)** computes an action for each vehicle from the features, outputting changes in speed and turning (with more actions added as training progresses).
4. Based on the reward function **f(sₜ, aₜ, sₜ₊₁)**, the reward is determined from the current state, next state, and chosen action.
5. The trajectory **(aₜ, sₜ, rₜ, sₜ₊₁)** is logged.  
   To save memory, only **aₜ and rₜ** and the initial state may be stored, since sₜ₊₁ can be reconstructed.
6. The **return** is calculated using the user-defined γ (gamma).
7. The **Value Network (VN)** predicts the return, giving an MSE-based value loss.
8. **Advantage** is computed as:  
   `advantage = return − VN_pred`.
9. The **clipped surrogate objective** then computes the expectation of the minimum between the clipped and unclipped terms (dependent on ε, advantage, and the ratio of old to new policies).
10. Combined losses are calculated using coefficients **c₁** and **c₂**, plus **entropy bonus** to encourage exploration.
11. **Backpropagation** updates the networks.
12. The episode steps forward one unit in time.
13. This repeats for hundreds of episodes until the first curriculum phase meets an acceptable standard.
14. Training continues through the phases until the final one is complete.
15. A **final evaluation** determines whether the result is strong enough.

---

## **Project Limitations & Purpose**

Because the available hardware may not have the computational strength required to reliably find a good minimum for both VN and PN losses, the agent may not fully converge. If the agent fails to understand the environment (i.e., does not converge to a “good” solution), we cannot definitively conclude whether this architecture can achieve intelligent behaviour under ideal training conditions.

The broader aim of this project is therefore to assess **whether a system like this—autonomous vehicles coordinated by a central controller—can outperform human-like or decentralised approaches**, and whether **this particular architecture** is:

- viable,
- efficient,
- scalable,
- or requiring improvement or replacement.

Ultimately, the result will show **how far PPO-based centralised control can go** within the constraints of this project, and where future investigation is needed.

---