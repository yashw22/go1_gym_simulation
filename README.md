# **Robot Navigation using Reinforcement Learning**

### Overview
- This project demonstrates autonomous navigation and obstacle avoidance within Nvidia's Isaac Gym Simulator.
- Using a model-free RL technique, the Unitree Go1 robot is trained to navigate towards a target while avoiding obstacles in an unknown flat environment.

---

### Key Features
- Builds upon the  [Improbable-AI/walk-these-ways repositrory](https://github.com/Improbable-AI/walk-these-ways), which provides a robust starter kit for RL-based robot locomotion.
- The base repository uses reinforcement learning to convert directional commands into joint values for the robot's legs, enabling locomotion.
- Extends the functionality of the base repository by enabling the robot to autonomously navigate to a specified target while avoiding obstacles.

---

### Methodology
Developed an RL model that:
- Utilizes the robot's current location, target location, and detected obstacle placements as inputs among other parameters.
- Generates directional commands for the robot, enabling autonomous movement and dynamic obstacle avoidance.
- **Simulator**: Utilized Nvidia's Isaac Gym, a high-performance physics simulator, to train and validate the model in a controlled virtual environment.
