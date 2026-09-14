# AGENTS.md — Reinforcement Learning Showcase (Godot)

This document provides a lightweight architectural overview of the scripts and core systems in this Godot reinforcement learning project for AI assistants and developers.

---

## Project Overview

This project is a 3D reinforcement learning showcase built in Godot. It implements tabular **Q-Learning** in a procedurally generated 12×12 Grid World where an agent must navigate around obstacles to collect items and optimize its path.

---

## Core Architecture & Scripts

All core logic resides in `scripts/`:

### 1. Orchestration & Control (`main.gd`)
- **Role**: Scene root node (`Node3D`). Manages game states, user input, simulation modes, and logging.
- **Modes**: Manual (`MANUAL`), Accelerated Training (`training_in_progress`), Trained Policy Playback (`TRAINED`), and Random Policy Playback (`RANDOM`).
- **Persistence**: Loads/saves Q-tables (`user://q_table.json`) and training logs (`res://data/training_log.json`).

### 2. Training & Evaluation (`trainer.gd`)
- **Role**: RefCounted class managing the RL training loop and evaluation.
- **Training**: Iterates through fixed training seeds (`[1, 2, 3, 4, 5, 6, 7, 8]`) with $\epsilon$-greedy exploration and exponential decay. Supports accelerated visual snapshots.
- **Evaluation**: Evaluates generalization on held-out test seeds (`[100, 101]`) with $\epsilon = 0.0$ and updates disabled.

### 3. Q-Learning Algorithm (`q_learning.gd`)
- **Role**: RefCounted implementation of tabular Q-learning.
- **Mechanics**:
  - State-action value table (`q_table: Dictionary`).
  - $\epsilon$-greedy action selection (`get_action`).
  - Bellman update rule with Temporal Difference (TD) error and learning rate $\alpha$, discount factor $\gamma$.
  - Versioned Q-table serialization (`STATE_FORMAT_VERSION := 2`) to prevent incompatible state mismatches.

### 4. Agent & State Representation (`agent.gd`)
- **Role**: 3D Node (`GridAgent`) representing the agent.
- **Movement & Rewards**: Handles grid collision, collectible pickup, smooth visual interpolation (`_process`), and reward shaping:
  - Step penalty: `-0.1`
  - Blocked action penalty: `-2.0`
  - Collectible reward: `+10.0`
  - Completion bonus: `+50.0`
- **State Key (`get_state_key`)**: Encodes the state string as:
  `[up_type, down_type, left_type, right_type, goal_direction, prev_action]`
  - *Goal Direction*: Computed via obstacle-aware Breadth-First Search (BFS) to find the first step along the shortest path to the nearest reachable collectible.
  - *Previous Action*: Prevents infinite back-and-forth oscillation loops.

### 5. Procedural Map Generation & Validation (`grid_world.gd` & `reachability.gd`)
- **Role**: Generates 12×12 maps with randomized obstacles and collectibles (5–8 items).
- **Reachability**: Uses BFS (`reachability.gd`) to validate that all collectibles are reachable from the agent start position before accepting a map seed.

### 6. Rendering (`grid_renderer.gd`)
- **Role**: Translates 2D logical grid arrays into 3D Godot `GridMap` tiles (floors, obstacles, collectibles) and computes agent world coordinates.

---

## Interactive Controls (Runtime)

| Key | Action |
| :--- | :--- |
| **W / A / S / D** | Manual movement (when in Manual mode) |
| **T** | Run Q-learning training |
| **P** | Playback trained policy |
| **R** | Playback random policy |
| **N** | Generate a new random map |
| **M** | Return to manual control mode |

---

## Key Highlights for College Project

1. **Robust State Engineering**: Combines local obstacle sensing, pathfinding-backed goal hints (BFS), and action history to eliminate deadlocks and loops.
2. **Generalization Testing**: Separates training seeds (1–8) from evaluation seeds (100–101) to demonstrate true policy generalization.
3. **Hybrid Performance**: Interleaves rapid headless/fast training episodes with periodic visual snapshots in Godot for smooth demonstration without sacrificing training speed.
