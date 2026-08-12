# Robot Skill Visualization and Interactive Demonstration System

This project is a simulation and interactive demonstration system designed for the **Unitree Go2 quadruped robot**. It is built with Genesis, PyTorch, Viser, Legged Gym, and RSL-RL.

The system loads pretrained reinforcement learning policies to perform dynamic skills, autonomous navigation, and basic motion control in the Genesis physics simulation environment. It also provides 3D visualization of the robot’s pose, motion state, sensor data, and planning results.

## 🤖 Core Features

### Dynamic Skill Demonstration

The system includes multiple policies trained for specific actions, enabling the Go2 robot to perform:

- Front-leg handstand;
- Cyclic rear-leg stand;
- Single backflip;
- Consecutive backflips;
- Forward spring jump.

These skills are not prerecorded animations. The system continuously reads the robot’s body orientation, angular velocity, joint positions, and joint velocities. The policy network then computes joint actions in real time.

The skill module also supports action transitions, landing detection, stabilization recovery, emergency stop, reset, replay, execution progress display, and scene screenshots.

### Autonomous Navigation

The autonomous navigation module combines environmental perception, path planning, and reinforcement learning-based locomotion into a complete pipeline:

```text
LiDAR Scanning
      ↓
Occupancy Grid Mapping
      ↓
Obstacle Inflation and Traversability Analysis
      ↓
A* Path Planning and Dynamic Replanning
      ↓
Local Target Points and Velocity Commands
      ↓
Reinforcement Learning Locomotion Policy
```

The navigation module supports 360° LiDAR scanning, occupancy grid updates, A* global planning, dynamic replanning, short-range obstacle repulsion, goal constraints, arrival detection, and visualization of paths, point clouds, and occupancy maps.

The system provides multiple navigation environments, including corridors with obstacle pillars, fenced areas, and open environments with scattered obstacles. When LiDAR is unavailable, the system can fall back to a static obstacle map.

### Motion Control

The project includes a pretrained `go2_wtw` locomotion policy that accepts high-level velocity and behavior commands to perform:

- Forward and backward movement;
- Left and right lateral movement;
- Left and right yaw rotation;
- Stop and standing reset;
- Motion command magnitude adjustment;
- Gait period and body height adjustment;
- Switching between camera tracking and free-view modes.

The system supports four basic gaits:

| Gait | Policy Name | Motion Characteristics |
|---|---|---|
| Diagonal trot | `trot` | Diagonal legs move in synchronization |
| Bounding gait | `bound` | Front and rear legs move in separate groups |
| Pacing gait | `pace` | Legs on the same side move alternately |
| Synchronized jump | `pronk` | All four legs move simultaneously |

## 🧠 Role of Reinforcement Learning

The reinforcement learning policy serves as the system’s low-level motion controller. It converts high-level task commands into actions for the Go2 robot’s 12 joints.

```text
Velocity Commands / Skill Commands
                ↓
Robot State and Observation History
                ↓
Pretrained Reinforcement Learning Policy
                ↓
Joint Position Targets or Control Torques
                ↓
Genesis Physics Simulation
```

Reinforcement learning is mainly used for two types of tasks:

1. **General locomotion control**: Coordinates all four legs according to forward velocity, lateral velocity, yaw velocity, and gait parameters.
2. **Specialized skill control**: Uses independent policies to perform dynamic actions such as handstands, rear-leg stands, backflips, and jumps.

Autonomous navigation is not implemented entirely through reinforcement learning. LiDAR handles environmental perception, A* performs path planning, and the reinforcement learning policy enables the robot to move stably in the direction specified by the planner.

During normal operation, the system directly loads existing models for inference and does not retrain the agents.

## 🏗️ System Components

| Module | Purpose |
|---|---|
| `skills` | Dynamic skill loading, execution, and state management |
| `nav` | LiDAR processing, occupancy grids, A* planning, and navigation control |
| `play` | Go2 locomotion policy, gait switching, and motion control |
| `common` | 3D visualization, robot meshes, camera control, status panels, and keyboard shortcuts |
| `weights` | Pretrained weights for locomotion, navigation, and dynamic skills |
| `assets` | Go2 simulation models and scene resources |
| `resources` | Go2 robot descriptions, URDF files, and mesh resources |
| `legged_gym` | Quadruped robot task environments and simulation interfaces |
| `rsl_rl` | Policy networks, PPO algorithms, runners, and data storage |
| `vendor` | Built-in scripts required for skill composition and navigation pipelines |

## 🔧 Technical Features

- Uses Genesis for rigid-body dynamics and joint control simulation;
- Uses PyTorch to load and execute reinforcement learning policies;
- Uses Viser to synchronize and display robot link poses and scene information;
- Supports both GPU and CPU computing backends;
- Stores models, weights, and robot resources within the project directory;
- Keeps the skill, navigation, and motion-control modules independent for separate execution and extension;
- Retains the core structures of Legged Gym and RSL-RL for future research and policy development.

## ⚠️ Scope of Use

This project is primarily intended for robot simulation research, algorithm validation, and skill demonstrations. The current control target is a virtual Go2 robot running in the Genesis environment.

The project does not currently integrate the Unitree real-robot SDK, ROS communication, or physical motor-control interfaces. Deployment on a physical robot would require additional communication interfaces, safety constraints, emergency-stop protection, state estimation, and Sim-to-Real adaptation.
