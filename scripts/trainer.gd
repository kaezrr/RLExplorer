extends RefCounted
class_name Trainer

var q_learning: QLearning
var agent: GridAgent
var grid_world: GridWorld

var episode_rewards: Array[float] = []

# Fixed training maps.
# These seeds are used for training only.
var training_seeds: Array[int] = [1, 2, 3, 4, 5, 6, 7, 8]

# Held-out evaluation maps.
# These seeds must never be used by run_training().
const TEST_SEEDS: Array[int] = [100, 101]

var max_steps: int = 150


func _init(q_learning_system: QLearning, agent_system: GridAgent, grid_world_system: GridWorld) -> void:
	q_learning = q_learning_system
	agent = agent_system
	grid_world = grid_world_system


func run_training(
	num_episodes: int,
	alpha: float,
	gamma: float,
	epsilon_start: float,
	epsilon_end: float,
	epsilon_decay: float,
	visual_interval: int = 50,
	visual_callback: Callable = Callable(),
) -> Array[float]:
	episode_rewards.clear()

	var epsilon := epsilon_start
	var safe_visual_interval: int = max(1, visual_interval)

	# Rewards for every episode since the last snapshot was shown.
	# The Q-learning runs every episode regardless of visual_interval,
	# so this batch is what lets the reward graph get one point per
	# real episode instead of one point per snapshot.
	var batch_rewards: Array[float] = []
	var batch_start_episode: int = 0

	for episode in range(num_episodes):
		# Cycle through the 8 training maps in round-robin order.
		var training_seed: int = training_seeds[episode % training_seeds.size()]

		# Only refresh the visible world periodically.
		# The actual Q-learning still runs every episode.
		var show_snapshot := (
			(episode + 1) % safe_visual_interval == 0 or episode == num_episodes - 1
		)

		# For snapshot episodes, capture the map's *starting* state -
		# full collectibles, agent at its spawn point - before a single
		# step runs. run_episode() below will mutate its own internally
		# generated grid as the episode plays out (often clearing every
		# collectible well within max_steps), so peeking the fresh map
		# here is what lets the snapshot show the "before" picture
		# instead of the "after" one.
		var snapshot_grid: Array = []
		var snapshot_start_position := Vector2i.ZERO
		var snapshot_collectible_count := 0

		if show_snapshot:
			snapshot_grid = grid_world.generate_map(training_seed)
			snapshot_start_position = grid_world.find_agent_start(snapshot_grid)

			for row in snapshot_grid:
				for cell in row:
					if cell == GridWorld.COLLECTIBLE:
						snapshot_collectible_count += 1

		var total_reward := run_episode(training_seed, alpha, gamma, epsilon)

		episode_rewards.append(total_reward)
		batch_rewards.append(total_reward)

		epsilon = max(epsilon_end, epsilon * epsilon_decay)

		if show_snapshot:
			if visual_callback.is_valid():
				visual_callback.call(
					{
						"episode": episode + 1,
						"total_episodes": num_episodes,
						"seed": training_seed,
						"reward": total_reward,
						"epsilon": epsilon,
						"remaining_collectibles": snapshot_collectible_count,
						"position": snapshot_start_position,
						"grid_snapshot": snapshot_grid,
						# All per-episode rewards since the last snapshot,
						# in order, starting at batch_start_episode + 1.
						"episode_rewards_batch": batch_rewards.duplicate(),
						"batch_start_episode": batch_start_episode + 1,
					}
				)

			batch_rewards.clear()
			batch_start_episode = episode + 1

			# Give Godot one frame to display this snapshot before
			# the next batch of episodes runs.
			await agent.get_tree().process_frame

			print(
				"Training | Episode ",
				episode + 1,
				"/",
				num_episodes,
				" | Seed: ",
				training_seed,
				" | Reward: ",
				total_reward,
				" | Epsilon: ",
				epsilon,
				" | Collectibles (before): ",
				snapshot_collectible_count,
				" | Start position: ",
				snapshot_start_position,
			)

	return episode_rewards


func run_episode(map_seed: int, alpha: float, gamma: float, epsilon: float) -> float:
	var grid := grid_world.generate_map(map_seed)
	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, agent.grid_renderer)

	var total_reward := 0.0

	for step in range(max_steps):
		var state_key := agent.get_state_key()

		var action := q_learning.get_action(state_key, epsilon)

		var result: Dictionary = agent.try_move(action)

		var reward: float = result.reward
		var next_state_key := agent.get_state_key()

		q_learning.update(state_key, action, reward, next_state_key, alpha, gamma)

		total_reward += reward

		if result.completed:
			break

	return total_reward


func evaluate_test_maps() -> Array[Dictionary]:
	var evaluation_results: Array[Dictionary] = []

	print("")
	print("Starting held-out evaluation")
	print("Evaluation seeds: ", TEST_SEEDS)
	print("Evaluation epsilon: 0.0")
	print("Q-table updates: disabled")
	print("")

	for test_seed in TEST_SEEDS:
		var result := evaluate_single_map(test_seed)
		evaluation_results.append(result)

		print(
			"Evaluation | Seed: ",
			test_seed,
			" | Reward: ",
			result["reward"],
			" | Steps: ",
			result["steps"],
			" | Completed: ",
			result["completed"],
		)

	print("")
	print("Held-out evaluation complete")
	print("")

	return evaluation_results


func evaluate_single_map(map_seed: int) -> Dictionary:
	var grid := grid_world.generate_map(map_seed)
	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, agent.grid_renderer)

	var total_reward := 0.0
	var steps_taken := 0
	var completed := false

	# epsilon = 0.0 means purely greedy evaluation.
	# No Q-learning update is performed here.
	for step in range(max_steps):
		var state_key := agent.get_state_key()

		var action := q_learning.get_action(state_key, 0.0)

		var result: Dictionary = agent.try_move(action)

		total_reward += float(result["reward"])
		steps_taken += 1

		if result["completed"]:
			completed = true
			break

	var remaining_collectibles := agent.get_collectible_count()

	return {
		"seed": map_seed,
		"reward": total_reward,
		"steps": steps_taken,
		"completed": completed,
		"remaining_collectibles": remaining_collectibles,
	}


func save_training_log(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)

	if file == null:
		print("Failed to open training log for writing: ", path)
		return false

	var json_text := JSON.stringify(episode_rewards)

	file.store_string(json_text)
	file.close()

	print("Training log saved: ", path)
	print("Episodes saved: ", episode_rewards.size())

	return true
