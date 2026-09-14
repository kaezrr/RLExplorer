extends Node3D

@onready var grid_map: GridMap = $GridMap
@onready var agent: GridAgent = $Agent
@onready var camera: Camera3D = $Camera3D


# --------------------------------------------------
# Map configuration
# --------------------------------------------------

# Current map seed.
#
# Training uses seeds 1-8.
# Held-out evaluation normally uses seed 100.
#
# Press N during the game to generate a new random map.
@export var MAP_SEED := 100

# --------------------------------------------------
# Training controls
# --------------------------------------------------

@export var training_episodes := 5000
@export var alpha := 0.1
@export var gamma := 0.9
@export var epsilon_start := 1.0
@export var epsilon_end := 0.05
@export var epsilon_decay := 0.995

@export var save_q_table_after_training := true
@export var load_trained_q_table := false


# --------------------------------------------------
# Accelerated visual training
# --------------------------------------------------

# Training still runs every episode, but the visible
# map is refreshed only every N episodes.
# This makes training feel fast while still letting
# you see the agent's current learned behaviour.
@export var visual_training_interval := 25

# --------------------------------------------------
# Q-table
# --------------------------------------------------

const Q_TABLE_PATH := "user://q_table.json"


# Prevents starting a second training run while one is
# already in progress.
var training_in_progress := false

# --------------------------------------------------
# Automatic playback configuration
# --------------------------------------------------

@export var playback_step_delay := 0.20

const MAX_PLAYBACK_STEPS := 150

enum PlaybackMode {
	TRAINED,
	RANDOM,
}

var playback_mode: PlaybackMode = PlaybackMode.TRAINED
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
# Initialization
# --------------------------------------------------


func _ready() -> void:
	grid_world = GridWorld.new()
	grid_renderer = GridRenderer.new()
	q_learning = QLearning.new()

	# Keep Trainer as part of the existing project architecture.
	trainer = Trainer.new(q_learning, agent, grid_world)

	add_child(grid_world)
	add_child(grid_renderer)

	# Build the initial map.
	setup_demo_map(MAP_SEED)

	print("")
	print("==========================================")
	print("RL GRID WORLD")
	print("==========================================")
	print("Current map seed: ", MAP_SEED)
	print("")
	print("Controls:")
	print("T = Train model")
	print("P = Trained policy playback")
	print("R = Random policy playback")
	print("N = New random map")
	print("")
	print("Q-table path: ", Q_TABLE_PATH)
	print("==========================================")
	print("")

	# --------------------------------------------------
	# Optional automatic training/loading.
	# --------------------------------------------------
	if load_trained_q_table:
		load_configured_q_table()

# --------------------------------------------------
# Map setup / reset
# --------------------------------------------------


func setup_demo_map(map_seed: int) -> void:
	# Stop playback.
	playback_running = false
	playback_timer = 0.0

	# Reset playback statistics.
	playback_steps = 0
	playback_total_reward = 0.0

	var grid := grid_world.generate_map(map_seed)

	grid_renderer.render_grid(grid, grid_map)

	var start_position := grid_world.find_agent_start(grid)

	agent.setup(start_position, grid, grid_renderer)

	playback_start_collectibles = (agent.get_collectible_count())

	print("")
	print("Map reset")
	print("Seed: ", map_seed)
	print("Start position: ", start_position)
	print("Collectibles: ", playback_start_collectibles)
	print("")

# --------------------------------------------------
# Generate a new random map
# --------------------------------------------------


func randomize_current_map() -> void:
	# Do not allow a map reset while training is running.
	if training_in_progress:
		print("")
		print("Cannot randomize map while training is in progress.")
		print("")
		return

	# Stop any automatic playback.
	playback_running = false
	playback_timer = 0.0

	# Generate a new random seed.
	var old_seed := MAP_SEED

	MAP_SEED = randi_range(1, 2147483647)

	# Make sure the new seed is actually different.
	if MAP_SEED == old_seed:
		MAP_SEED += 1

	print("")
	print("==========================================")
	print("NEW RANDOM MAP")
	print("==========================================")
	print("Previous seed: ", old_seed)
	print("New seed: ", MAP_SEED)
	print("==========================================")
	print("")

	# Build the new map.
	setup_demo_map(MAP_SEED)

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
	if training_in_progress:
		print("Training already in progress, ignoring request.")

		return

	if playback_running:
		print("Cannot start training while playback is running.")

		return

	training_in_progress = true
	playback_running = false

	print("")
	print("==========================================")
	print("STARTING ACCELERATED VISUAL TRAINING")
	print("==========================================")
	print("Episodes: ", training_episodes)
	print("Training seeds: ", trainer.training_seeds)
	print("Alpha: ", alpha)
	print("Gamma: ", gamma)
	print("Epsilon start: ", epsilon_start)
	print("Epsilon end: ", epsilon_end)
	print("Epsilon decay: ", epsilon_decay)
	print("Visual map refresh: every ", visual_training_interval, " episodes")
	print("")
	print("The agent is training rapidly between visual snapshots.")
	print("P / R / M / N are disabled until training finishes.")
	print("")

	# run_training() still performs the exact same Q-learning
	# update on every episode. The only difference is that
	# every N episodes it gives Godot one frame to display
	# the current training state.
	var rewards := await trainer.run_training(
		training_episodes,
		alpha,
		gamma,
		epsilon_start,
		epsilon_end,
		epsilon_decay,
		visual_training_interval,
		Callable(self, "_on_training_snapshot"),
	)

	var log_saved := trainer.save_training_log("res://data/training_log.json")

	var q_table_saved := false

	if save_q_table_after_training:
		q_table_saved = q_learning.save_q_table(Q_TABLE_PATH)

	print("")
	print("==========================================")
	print("TRAINING COMPLETE")
	print("==========================================")
	print("Episodes completed: ", rewards.size())
	print("Q-table states learned: ", q_learning.q_table.size())
	print("Training log save successful: ", log_saved)
	print("Q-table save successful: ", q_table_saved)
	print("")

	# --------------------------------------------------
	# Held-out evaluation.
	# --------------------------------------------------
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
			result["remaining_collectibles"],
		)

	print("")

	training_in_progress = false

	setup_demo_map(MAP_SEED)

	print("")
	print("Training finished.")
	print("P = trained playback")
	print("R = random playback")
	print("N = new random map")
	print("")

# --------------------------------------------------
# Visual training snapshot
# --------------------------------------------------


func _on_training_snapshot(data: Dictionary) -> void:
	# The Trainer has just finished an actual training
	# episode. Its agent contains the final state of that
	# episode, including any collectibles it removed.
	# Render that state now so the player can see how the
	# policy is behaving.
	grid_renderer.render_grid(agent.grid_data, grid_map)

	print(
		"Training snapshot | Episode ",
		data["episode"],
		"/",
		data["total_episodes"],
		" | Seed: ",
		data["seed"],
		" | Reward: ",
		data["reward"],
		" | Epsilon: ",
		data["epsilon"],
		" | Remaining: ",
		data["remaining_collectibles"],
		" | Agent position: ",
		data["position"],
	)

# --------------------------------------------------
# Start trained playback
# --------------------------------------------------


func start_trained_playback() -> void:
	if training_in_progress:
		print("Cannot start playback while training is in progress.")

		return

	print("")
	print("==========================================")
	print("TRAINED POLICY PLAYBACK")
	print("==========================================")
	print("Current map seed: ", MAP_SEED)

	# Load the saved policy before starting playback.
	if not load_configured_q_table():
		print("ERROR: Could not load trained Q-table.")

		playback_running = false

		return

	# Reset to the current map.
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
	if training_in_progress:
		print("Cannot start playback while training is in progress.")

		return

	print("")
	print("==========================================")
	print("RANDOM POLICY PLAYBACK")
	print("==========================================")
	print("Current map seed: ", MAP_SEED)

	# Reset to the current map.
	setup_demo_map(MAP_SEED)

	playback_mode = PlaybackMode.RANDOM
	playback_running = true
	playback_timer = playback_step_delay

	print("Policy: Random")
	print("Playback started")
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

	# --------------------------------------------------
	# Trained policy.
	# --------------------------------------------------
	if playback_mode == PlaybackMode.TRAINED:
		action = q_learning.get_action(state_key, 0.0)

	# --------------------------------------------------
	# Random policy.
	#
	# This does not modify the Q-table.
	# --------------------------------------------------

	elif playback_mode == PlaybackMode.RANDOM:
		action = randi_range(GridAgent.Action.UP, GridAgent.Action.RIGHT)

	else:
		return

	var result: Dictionary = agent.try_move(action)

	var reward: float = result["reward"]

	playback_total_reward += reward
	playback_steps += 1

	# --------------------------------------------------
	# Remove collected item visually.
	# --------------------------------------------------
	if result["collected"]:
		grid_renderer.render_grid(agent.grid_data, grid_map)

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
		agent.get_collectible_count(),
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
		playback_start_collectibles,
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
		# Train model
		# ----------------------------------------------
		KEY_T:
			run_configured_training()

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
		# New random map
		# ----------------------------------------------
		KEY_N:
			randomize_current_map()

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
