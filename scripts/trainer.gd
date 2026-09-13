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


func _init(
	q_learning_system: QLearning,
	agent_system: GridAgent,
	grid_world_system: GridWorld
) -> void:
	q_learning = q_learning_system
	agent = agent_system
	grid_world = grid_world_system


func run_training(
	num_episodes: int,
	alpha: float,
	gamma: float,
	epsilon_start: float,
	epsilon_end: float,
	epsilon_decay: float
) -> Array[float]:
	episode_rewards.clear()

	var epsilon := epsilon_start

	for episode in range(num_episodes):
		# Cycle through the 8 training maps in round-robin order.
		var training_seed: int = training_seeds[episode % training_seeds.size()]

		var total_reward := run_episode(
			training_seed,
			alpha,
			gamma,
			epsilon
		)

		episode_rewards.append(total_reward)

		epsilon = max(epsilon_end, epsilon * epsilon_decay)

		print(
			"Episode ",
			episode + 1,
			"/",
			num_episodes,
			" | Seed: ",
			training_seed,
			" | Reward: ",
			total_reward,
			" | Epsilon: ",
			epsilon
		)

	return episode_rewards


func run_episode(
	map_seed: int,
	alpha: float,
	gamma: float,
	epsilon: float
) -> float:
	var grid := grid_world.generate_map(map_seed)
	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, agent.grid_renderer)

	var total_reward := 0.0

	for step in range(max_steps):
		var state_key := agent.get_state_key()

		var action := q_learning.get_action(
			state_key,
			epsilon
		)

		var result: Dictionary = agent.try_move(action)

		var reward: float = result.reward
		var next_state_key := agent.get_state_key()

		q_learning.update(
			state_key,
			action,
			reward,
			next_state_key,
			alpha,
			gamma
		)

		total_reward += reward

		if result.completed:
			break

	return total_reward


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
