extends RefCounted
class_name Trainer

## V3 training/evaluation orchestration.
##
## This class keeps the existing Main -> Trainer architecture, but changes
## the experiment from a small fixed set of memorized maps to fresh random
## maps every episode, matching the Python V3 trainer.

const GRID_W := 12
const GRID_H := 12
const NUM_COLLECTIBLES := 6
const MAX_STEPS := 1152

const TRAIN_EPISODES_DEFAULT := 15000
const GREEDY_FINETUNE_EPISODES_DEFAULT := 5000
const TEST_EPISODES_DEFAULT := 300

const ALPHA_DEFAULT := 0.015
const GAMMA_DEFAULT := 0.98
const EPSILON_START_DEFAULT := 1.0
const EPSILON_MIN_DEFAULT := 0.05
const EPSILON_DECAY_DEFAULT := 0.9995

const RANDOM_SEED := 12345

# Training/evaluation seed spaces are deliberately disjoint.
const TEST_SEED_OFFSET := 2000000
const RANDOM_BASELINE_SEED_OFFSET := 1000000

var q_learning: QLearning
var agent: GridAgent
var grid_world: GridWorld

var episode_rewards: Array[float] = []
var episode_success: Array[bool] = []
var episode_items: Array[int] = []
var episode_steps: Array[int] = []
var episode_coverage: Array[float] = []

# Compatibility fields retained so existing Main/UI code that prints or reads
# these values does not break while we transition the implementation.
var training_seeds: Array[int] = []
const TEST_SEEDS: Array[int] = []
var max_steps: int = MAX_STEPS


func _init(
	q_learning_system: QLearning,
	agent_system: GridAgent,
	grid_world_system: GridWorld,
) -> void:
	q_learning = q_learning_system
	agent = agent_system
	grid_world = grid_world_system


func run_training(
	num_episodes: int,
	alpha: float = ALPHA_DEFAULT,
	gamma: float = GAMMA_DEFAULT,
	epsilon_start: float = EPSILON_START_DEFAULT,
	epsilon_end: float = EPSILON_MIN_DEFAULT,
	epsilon_decay: float = EPSILON_DECAY_DEFAULT,
	visual_interval: int = 50,
	visual_callback: Callable = Callable(),
	greedy_finetune_episodes: int = 0,
) -> Array[float]:
	episode_rewards.clear()
	episode_success.clear()
	episode_items.clear()
	episode_steps.clear()
	episode_coverage.clear()

	var epsilon := epsilon_start
	var safe_visual_interval: int = max(1, visual_interval)
	var safe_finetune := clampi(greedy_finetune_episodes, 0, num_episodes)
	var finetune_start := num_episodes - safe_finetune

	var batch_rewards: Array[float] = []
	var batch_start_episode := 0

	for episode in range(num_episodes):
		# Fresh random map every episode. This is the core V3 generalization
		# change versus the old eight-map round-robin trainer.
		var map_seed := RANDOM_SEED + episode
		var show_snapshot := (
			(episode + 1) % safe_visual_interval == 0
			or episode == num_episodes - 1
		)

		var snapshot_grid: Array = []
		var snapshot_start_position := Vector2i.ZERO
		var snapshot_collectibles := 0

		if show_snapshot:
			snapshot_grid = grid_world.generate_map(map_seed)
			snapshot_start_position = grid_world.find_agent_start(snapshot_grid)
			snapshot_collectibles = grid_world.get_collectibles(snapshot_grid).size()

		# V3's final phase is pure greedy fine-tuning. This is applied to the
		# action-selection epsilon, while TD learning continues normally.
		if episode >= finetune_start:
			epsilon = 0.0

		var result := run_episode(map_seed, alpha, gamma, epsilon)

		episode_rewards.append(float(result["reward"]))
		episode_success.append(bool(result["success"]))
		episode_items.append(int(result["items"]))
		episode_steps.append(int(result["steps"]))
		episode_coverage.append(float(result["coverage"]))
		batch_rewards.append(float(result["reward"]))

		# Match the Python trainer's decay call after every episode. Once the
		# greedy fine-tune phase starts epsilon is explicitly held at zero.
		if episode < finetune_start:
			epsilon = max(epsilon_end, epsilon * epsilon_decay)
		else:
			epsilon = 0.0

		if show_snapshot and visual_callback.is_valid():
			visual_callback.call({
				"episode": episode + 1,
				"total_episodes": num_episodes,
				"seed": map_seed,
				"reward": float(result["reward"]),
				"epsilon": epsilon,
				"remaining_collectibles": snapshot_collectibles,
				"position": snapshot_start_position,
				"grid_snapshot": snapshot_grid,
				"episode_rewards_batch": batch_rewards.duplicate(),
				"batch_start_episode": batch_start_episode + 1,
				"success": bool(result["success"]),
				"items": int(result["items"]),
				"steps": int(result["steps"]),
				"coverage": float(result["coverage"]),
				"weights": q_learning.get_weights(),
				"features": QLearning.FEATURE_COUNT,
			})

			batch_rewards.clear()
			batch_start_episode = episode + 1

			# Yield a frame so the existing Godot visualization remains visible
			# between accelerated training batches.
			await agent.get_tree().process_frame

			print(
				"Training | Episode ",
				episode + 1,
				"/",
				num_episodes,
				" | Seed: ",
				map_seed,
				" | Reward: ",
				float(result["reward"]),
				" | Epsilon: ",
				epsilon,
				" | Items: ",
				int(result["items"]),
				"/",
				NUM_COLLECTIBLES,
				" | Success: ",
				bool(result["success"]),
				" | Steps: ",
				int(result["steps"]),
				" | Coverage: ",
				float(result["coverage"]) * 100.0,
				"%",
			)

	return episode_rewards


func run_episode(
	map_seed: int,
	alpha: float = ALPHA_DEFAULT,
	gamma: float = GAMMA_DEFAULT,
	epsilon: float = EPSILON_START_DEFAULT,
) -> Dictionary:
	var grid := grid_world.generate_map(map_seed)
	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, agent.grid_renderer, gamma)

	var total_reward := 0.0
	var final_result: Dictionary = {}

	for _step in range(max_steps):
		var phi := agent.get_features()
		var action := q_learning.get_action(phi, epsilon)

		var result: Dictionary = agent.try_move(action, gamma)
		var reward := float(result["reward"])
		var done := bool(result["done"])
		var next_phi := agent.get_features()

		q_learning.update(phi, action, reward, next_phi, done, alpha, gamma)
		total_reward += reward
		final_result = result

		if done:
			break

	return {
		"reward": total_reward,
		"steps": agent.steps,
		"success": agent.get_collectible_count() == 0,
		"items": agent.get_items_collected(),
		"coverage": agent.get_coverage(),
		"new_cells": agent.new_cells,
		"backtracks": agent.backtracks,
		"wall_hits": agent.wall_hits,
		"grid": grid,
		"start": start_position,
		"collectibles": grid_world.get_collectibles(grid),
		"trajectory": agent.trajectory.duplicate(),
		"last_result": final_result,
	}


func evaluate_test_maps(
	episodes: int = TEST_EPISODES_DEFAULT,
	seed_offset: int = TEST_SEED_OFFSET,
) -> Array[Dictionary]:
	var results: Array[Dictionary] = []

	print("")
	print("============================================================")
	print("LEARNED AGENT EVALUATION (UNSEEN RANDOM MAPS)")
	print("============================================================")
	print("Episodes: ", episodes)
	print("Seed range: ", RANDOM_SEED + seed_offset, " ... ", RANDOM_SEED + seed_offset + episodes - 1)
	print("Epsilon: 0.0")
	print("Learning updates: disabled")
	print("")

	for i in range(episodes):
		var seed := RANDOM_SEED + seed_offset + i
		var result := evaluate_single_map(seed)
		results.append(result)

	print_evaluation_summary("LEARNED AGENT (UNSEEN MAPS)", results)
	return results


func evaluate_single_map(map_seed: int) -> Dictionary:
	var grid := grid_world.generate_map(map_seed)
	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, agent.grid_renderer, GAMMA_DEFAULT)

	var total_reward := 0.0
	var steps_taken := 0
	var done := false

	for _step in range(max_steps):
		var phi := agent.get_features()
		var action := q_learning.get_action(phi, 0.0)
		var result: Dictionary = agent.try_move(action, GAMMA_DEFAULT)

		total_reward += float(result["reward"])
		steps_taken += 1
		done = bool(result["done"])

		# No q_learning.update() call here: V3 evaluation is purely greedy.
		if done:
			break

	return {
		"seed": map_seed,
		"reward": total_reward,
		"steps": steps_taken,
		"success": agent.get_collectible_count() == 0,
		"completed": agent.get_collectible_count() == 0,
		"items": agent.get_items_collected(),
		"remaining_collectibles": agent.get_collectible_count(),
		"coverage": agent.get_coverage(),
		"backtracks": agent.backtracks,
		"wall_hits": agent.wall_hits,
		"grid": grid,
		"start": start_position,
		"collectibles": grid_world.get_collectibles(grid),
		"trajectory": agent.trajectory.duplicate(),
	}


func evaluate_random_baseline(
	episodes: int = TEST_EPISODES_DEFAULT,
	seed_offset: int = RANDOM_BASELINE_SEED_OFFSET,
) -> Array[Dictionary]:
	var results: Array[Dictionary] = []

	for i in range(episodes):
		var seed := RANDOM_SEED + seed_offset + i
		var grid := grid_world.generate_map(seed)
		var start_position := grid_world.find_agent_start(grid)

		agent.setup(start_position, grid, agent.grid_renderer, GAMMA_DEFAULT)

		var total_reward := 0.0
		var steps_taken := 0
		var done := false

		for _step in range(max_steps):
			var action := q_learning.rng.randi_range(0, QLearning.ACTION_COUNT - 1)
			var result: Dictionary = agent.try_move(action, GAMMA_DEFAULT)

			total_reward += float(result["reward"])
			steps_taken += 1
			done = bool(result["done"])

			if done:
				break

		results.append({
			"seed": seed,
			"reward": total_reward,
			"steps": steps_taken,
			"success": agent.get_collectible_count() == 0,
			"completed": agent.get_collectible_count() == 0,
			"items": agent.get_items_collected(),
			"remaining_collectibles": agent.get_collectible_count(),
			"coverage": agent.get_coverage(),
			"backtracks": agent.backtracks,
			"wall_hits": agent.wall_hits,
		})

	print_evaluation_summary("RANDOM BASELINE", results)
	return results


func print_evaluation_summary(name: String, results: Array[Dictionary]) -> void:
	if results.is_empty():
		print(name, ": no results")
		return

	var reward_sum := 0.0
	var steps_sum := 0.0
	var item_sum := 0.0
	var coverage_sum := 0.0
	var backtrack_sum := 0.0
	var wall_sum := 0.0
	var success_count := 0

	for result in results:
		reward_sum += float(result["reward"])
		steps_sum += float(result["steps"])
		item_sum += float(result["items"])
		coverage_sum += float(result["coverage"])
		backtrack_sum += float(result["backtracks"])
		wall_sum += float(result["wall_hits"])
		if bool(result["success"]):
			success_count += 1

	var count := float(results.size())

	print("")
	print("============================================================")
	print(name)
	print("============================================================")
	print("Success rate:     ", (float(success_count) / count) * 100.0, "%")
	print("Avg items:        ", item_sum / count, " / ", NUM_COLLECTIBLES)
	print("Avg reward:       ", reward_sum / count)
	print("Avg steps:        ", steps_sum / count)
	print("Avg coverage:     ", (coverage_sum / count) * 100.0, "%")
	print("Avg backtracks:   ", backtrack_sum / count)
	print("Avg wall hits:    ", wall_sum / count)
	print("============================================================")


func get_training_summary() -> Dictionary:
	if episode_rewards.is_empty():
		return {
			"episodes": 0,
			"avg_reward": 0.0,
			"success_rate": 0.0,
			"avg_items": 0.0,
			"avg_steps": 0.0,
			"avg_coverage": 0.0,
		}

	var count := float(episode_rewards.size())
	var reward_sum := 0.0
	var item_sum := 0.0
	var steps_sum := 0.0
	var coverage_sum := 0.0
	var success_count := 0

	for i in range(episode_rewards.size()):
		reward_sum += episode_rewards[i]
		item_sum += episode_items[i]
		steps_sum += episode_steps[i]
		coverage_sum += episode_coverage[i]
		if episode_success[i]:
			success_count += 1

	return {
		"episodes": episode_rewards.size(),
		"avg_reward": reward_sum / count,
		"success_rate": float(success_count) / count,
		"avg_items": item_sum / count,
		"avg_steps": steps_sum / count,
		"avg_coverage": coverage_sum / count,
	}


func save_training_log(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		print("Failed to open training log for writing: ", path)
		return false

	var payload := {
		"version": 3,
		"random_seed": RANDOM_SEED,
		"episode_rewards": episode_rewards,
		"episode_success": episode_success,
		"episode_items": episode_items,
		"episode_steps": episode_steps,
		"episode_coverage": episode_coverage,
		"summary": get_training_summary(),
	}

	file.store_string(JSON.stringify(payload))
	file.close()

	print("Training log saved: ", path)
	print("Episodes saved: ", episode_rewards.size())
	return true
