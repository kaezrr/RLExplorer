extends RefCounted
class_name Trainer


var q_learning: QLearning
var agent: GridAgent
var grid_world: GridWorld

var episode_rewards: Array[float] = []

var training_seed: int = 42
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

		var total_reward := run_episode(
			alpha,
			gamma,
			epsilon
		)

		episode_rewards.append(total_reward)

		epsilon = max(
			epsilon_end,
			epsilon * epsilon_decay
		)

		print(
			"Episode ",
			episode + 1,
			"/",
			num_episodes,
			" | Reward: ",
			total_reward,
			" | Epsilon: ",
			epsilon
		)

	return episode_rewards


func run_episode(
	alpha: float,
	gamma: float,
	epsilon: float
) -> float:

	# --------------------------------------------------
	# Generate/reset the training map.
	# --------------------------------------------------

	var grid := grid_world.generate_map(
		training_seed
	)

	var start_position := grid_world.find_agent_start(
		grid
	)

	agent.setup(
		start_position,
		grid,
		agent.grid_renderer
	)

	# --------------------------------------------------
	# Run one episode.
	# --------------------------------------------------

	var total_reward := 0.0

	for step in range(max_steps):

		# 1. Observe current state.
		var state_key := agent.get_state_key()

		# 2. Choose action using epsilon-greedy policy.
		var action := q_learning.get_action(
			state_key,
			epsilon
		)

		# 3. Take the action in the environment.
		var result: Dictionary = agent.try_move(
			action
		)

		# 4. Receive reward.
		var reward: float = result.reward

		# 5. Observe the new state.
		var next_state_key := agent.get_state_key()

		# 6. Update Q-table.
		q_learning.update(
			state_key,
			action,
			reward,
			next_state_key,
			alpha,
			gamma
		)

		# 7. Accumulate episode reward.
		total_reward += reward

		# 8. Stop if all collectibles were collected.
		if result.completed:
			break

	return total_reward
