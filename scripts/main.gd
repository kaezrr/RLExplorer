extends Node3D

@onready var grid_map: GridMap = $GridMap
@onready var agent: GridAgent = $Agent
@onready var camera: Camera3D = $Camera3D

# --------------------------------------------------
# Map configuration
# --------------------------------------------------

# Held-out demo seed.
# This seed is intentionally separate from training seeds 1-8.
@export var MAP_SEED := 100

# --------------------------------------------------
# Training controls
# --------------------------------------------------

@export var run_training := false
@export var training_episodes := 5000
@export var alpha := 0.1
@export var gamma := 0.9
@export var epsilon_start := 1.0
@export var epsilon_end := 0.05
@export var epsilon_decay := 0.995

@export var save_q_table_after_training := true
@export var load_trained_q_table := false

const Q_TABLE_PATH := "user://q_table.json"

# --------------------------------------------------
# Automatic playback configuration
# --------------------------------------------------

# Time between automatic agent actions.
# Lower = faster playback.
@export var playback_step_delay := 0.20

const MAX_PLAYBACK_STEPS := 150

enum PlaybackMode {
	MANUAL,
	TRAINED,
	RANDOM
}

var playback_mode: PlaybackMode = PlaybackMode.MANUAL
var playback_running := false

var playback_timer := 0.0
var playback_steps := 0
var playback_total_reward := 0.0
var playback_start_collectibles := 0

# --------------------------------------------------
# Core systems
# --------------------------------------------------

var grid_world: GridWorld
var grid_renderer: GridRenderer
var q_learning: QLearning
var trainer: Trainer


# --------------------------------------------------
# Camera
# --------------------------------------------------

func position_camera(grid_size: int) -> void:
	var center := Vector3(grid_size / 2.0, 0, grid_size / 2.0)
	var horizontal_distance := grid_size * 0.9
	var height := grid_size * 1.6

	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = grid_size * 1.15

	camera.position = center + Vector3(
		horizontal_distance,
		height,
		horizontal_distance
	)

	camera.look_at(center, Vector3.UP)


# --------------------------------------------------
# Initialization
# --------------------------------------------------

func _ready() -> void:
	grid_world = GridWorld.new()
	grid_renderer = GridRenderer.new()
	q_learning = QLearning.new()

	# Keep Trainer as part of the existing project architecture.
	# Training and evaluation continue to use the same systems.
	trainer = Trainer.new(
		q_learning,
		agent,
		grid_world
	)

	add_child(grid_world)
	add_child(grid_renderer)

	# Build the initial held-out demo map.
	setup_demo_map(MAP_SEED)

	position_camera(GridWorld.GRID_SIZE)

	print("")
	print("==========================================")
	print("RL GRID WORLD")
	print("==========================================")
	print("Held-out demo seed: ", MAP_SEED)
	print("")
	print("Controls:")
	print("W / A / S / D = Manual movement")
	print("P = Trained policy playback")
	print("R = Random policy playback")
	print("M = Manual control")
	print("")
	print("Q-table path: ", Q_TABLE_PATH)
	print("==========================================")
	print("")


# --------------------------------------------------
# Map setup / reset
# --------------------------------------------------

func setup_demo_map(map_seed: int) -> void:
	playback_running = false
	playback_timer = 0.0
	playback_steps = 0
	playback_total_reward = 0.0

	var grid := grid_world.generate_map(map_seed)

	grid_renderer.render_grid(
		grid,
		grid_map
	)

	var start_position := grid_world.find_agent_start(grid)

	agent.setup(
		start_position,
		grid,
		grid_renderer
	)

	playback_start_collectibles = agent.get_collectible_count()

	print("")
	print("Demo map reset")
	print("Seed: ", map_seed)
	print("Start position: ", start_position)
	print("Collectibles: ", playback_start_collectibles)
	print("")


# --------------------------------------------------
# Q-table loading
# --------------------------------------------------

func load_configured_q_table() -> bool:
	print("")
	print("Loading trained Q-table")
	print("Path: ", Q_TABLE_PATH)

	var loaded := q_learning.load_q_table(Q_TABLE_PATH)

	print("Q-table load successful: ", loaded)

	if loaded:
		print("Q-table states: ", q_learning.q_table.size())

	print("")

	return loaded


# --------------------------------------------------
# Training
# --------------------------------------------------

func run_configured_training() -> void:
	print("")
	print("Starting configured training")
	print("Episodes: ", training_episodes)
	print("Training seeds: ", trainer.training_seeds)
	print("Alpha: ", alpha)
	print("Gamma: ", gamma)
	print("Epsilon start: ", epsilon_start)
	print("Epsilon end: ", epsilon_end)
	print("Epsilon decay: ", epsilon_decay)
	print("")

	var rewards := trainer.run_training(
		training_episodes,
		alpha,
		gamma,
		epsilon_start,
		epsilon_end,
		epsilon_decay
	)

	var log_saved := trainer.save_training_log(
		"res://data/training_log.json"
	)

	var q_table_saved := false

	if save_q_table_after_training:
		q_table_saved = q_learning.save_q_table(
			Q_TABLE_PATH
		)

	print("")
	print("Training complete")
	print("Episodes completed: ", rewards.size())
	print("Q-table states learned: ", q_learning.q_table.size())
	print("Training log save successful: ", log_saved)
	print("Q-table save successful: ", q_table_saved)
	print("")

	var evaluation_results := trainer.evaluate_test_maps()

	print("HELD-OUT EVALUATION RESULTS")

	for result in evaluation_results:
		print(
			"Seed: ",
			result["seed"],
			" | Reward: ",
			result["reward"],
			" | Steps: ",
			result["steps"],
			" | Completed: ",
			result["completed"],
			" | Remaining collectibles: ",
			result["remaining_collectibles"]
		)

	print("")


# --------------------------------------------------
# Start trained playback
# --------------------------------------------------

func start_trained_playback() -> void:
	print("")
	print("==========================================")
	print("TRAINED POLICY PLAYBACK")
	print("==========================================")
	print("Held-out seed: ", MAP_SEED)

	# Load the saved policy before starting playback.
	if not load_configured_q_table():
		print("ERROR: Could not load trained Q-table.")
		print("Returning to manual control.")
		playback_mode = PlaybackMode.MANUAL
		return

	# Reset to the exact same held-out map used by Random mode.
	setup_demo_map(MAP_SEED)

	playback_mode = PlaybackMode.TRAINED
	playback_running = true
	playback_timer = playback_step_delay

	print("Policy: Trained")
	print("Epsilon: 0.0")
	print("Playback started")
	print("")


# --------------------------------------------------
# Start random playback
# --------------------------------------------------

func start_random_playback() -> void:
	print("")
	print("==========================================")
	print("RANDOM POLICY PLAYBACK")
	print("==========================================")
	print("Held-out seed: ", MAP_SEED)

	# Reset to the exact same map seed used by trained playback.
	setup_demo_map(MAP_SEED)

	playback_mode = PlaybackMode.RANDOM
	playback_running = true
	playback_timer = playback_step_delay

	print("Policy: Random")
	print("Epsilon: 1.0")
	print("Playback started")
	print("")


# --------------------------------------------------
# Return to manual mode
# --------------------------------------------------

func start_manual_mode() -> void:
	playback_running = false
	playback_mode = PlaybackMode.MANUAL
	playback_timer = 0.0

	print("")
	print("Manual control enabled")
	print("Use W/A/S/D to move the agent.")
	print("")


# --------------------------------------------------
# Automatic playback loop
# --------------------------------------------------

func _process(delta: float) -> void:
	if not playback_running:
		return

	playback_timer -= delta

	if playback_timer > 0.0:
		return

	playback_timer = playback_step_delay

	run_playback_step()


# --------------------------------------------------
# Execute one automatic action
# --------------------------------------------------

func run_playback_step() -> void:
	if not playback_running:
		return

	# Safety termination condition.
	if playback_steps >= MAX_PLAYBACK_STEPS:
		finish_playback(false)
		return

	var state_key := agent.get_state_key()

	var action: int

	if playback_mode == PlaybackMode.TRAINED:
		# Pure exploitation.
		# epsilon = 0 means the trained policy always chooses
		# the highest-Q action.
		action = q_learning.get_action(
			state_key,
			0.0
		)

	elif playback_mode == PlaybackMode.RANDOM:
		# Random baseline.
		#
		# We deliberately generate the random action directly instead
		# of calling q_learning.get_action(), because the random
		# baseline should not modify the Q-table.
		action = randi_range(
			0,
			GridAgent.Action.RIGHT
		)

	else:
		return

	var result: Dictionary = agent.try_move(action)

	var reward: float = result["reward"]

	playback_total_reward += reward
	playback_steps += 1

	# A collectible is removed from agent.grid_data by try_move().
	# Re-rendering here removes its visual representation from GridMap.
	if result["collected"]:
		grid_renderer.render_grid(
			agent.grid_data,
			grid_map
		)

	print(
		"Playback step ",
		playback_steps,
		" | State: ",
		state_key,
		" | Action: ",
		action_name(action),
		" | Reward: ",
		reward,
		" | Total reward: ",
		playback_total_reward,
		" | Remaining collectibles: ",
		agent.get_collectible_count()
	)

	if result["completed"]:
		finish_playback(true)


# --------------------------------------------------
# Finish automatic episode
# --------------------------------------------------

func finish_playback(completed: bool) -> void:
	playback_running = false

	var policy_name := "Unknown"

	if playback_mode == PlaybackMode.TRAINED:
		policy_name = "Trained"

	elif playback_mode == PlaybackMode.RANDOM:
		policy_name = "Random"

	print("")
	print("==========================================")
	print("PLAYBACK COMPLETE")
	print("==========================================")
	print("Policy: ", policy_name)
	print("Seed: ", MAP_SEED)
	print("Steps: ", playback_steps)
	print("Reward: ", playback_total_reward)
	print(
		"Items collected: ",
		playback_start_collectibles - agent.get_collectible_count(),
		"/",
		playback_start_collectibles
	)
	print("Remaining collectibles: ", agent.get_collectible_count())
	print("Completed: ", completed)
	print("==========================================")
	print("")


# --------------------------------------------------
# Input
# --------------------------------------------------

func _unhandled_input(event: InputEvent) -> void:
	if not event is InputEventKey:
		return

	if not event.pressed or event.echo:
		return

	match event.keycode:

		# ----------------------------------------------
		# Manual movement
		# ----------------------------------------------

		KEY_W:
			if playback_mode == PlaybackMode.MANUAL:
				test_agent_move(GridAgent.Action.UP)

		KEY_S:
			if playback_mode == PlaybackMode.MANUAL:
				test_agent_move(GridAgent.Action.DOWN)

		KEY_A:
			if playback_mode == PlaybackMode.MANUAL:
				test_agent_move(GridAgent.Action.LEFT)

		KEY_D:
			if playback_mode == PlaybackMode.MANUAL:
				test_agent_move(GridAgent.Action.RIGHT)

		# ----------------------------------------------
		# Trained policy
		# ----------------------------------------------

		KEY_P:
			start_trained_playback()

		# ----------------------------------------------
		# Random policy
		# ----------------------------------------------

		KEY_R:
			start_random_playback()

		# ----------------------------------------------
		# Manual mode
		# ----------------------------------------------

		KEY_M:
			start_manual_mode()


# --------------------------------------------------
# Manual movement
# --------------------------------------------------

func test_agent_move(action: int) -> void:
	var result := agent.try_move(action)

	# If the manual move collected an item, refresh the
	# GridMap so the collectible disappears visually.
	if result["collected"]:
		grid_renderer.render_grid(
			agent.grid_data,
			grid_map
		)

	print(
		"Manual action: ",
		action_name(action),
		" | Result: ",
		result,
		" | State: ",
		agent.get_state_key()
	)


# --------------------------------------------------
# Utility
# --------------------------------------------------

func action_name(action: int) -> String:
	match action:
		GridAgent.Action.UP:
			return "UP"

		GridAgent.Action.DOWN:
			return "DOWN"

		GridAgent.Action.LEFT:
			return "LEFT"

		GridAgent.Action.RIGHT:
			return "RIGHT"

	return "UNKNOWN"
