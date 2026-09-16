extends Node3D
class_name RLExplorerMain

@onready var grid_map: GridMap = $GridMap
@onready var agent: GridAgent = $Agent
@onready var camera: Camera3D = $Camera3D
@onready var hud: HUD = $UI

# --------------------------------------------------
# V3 map configuration
# --------------------------------------------------

@export var MAP_SEED := 100

# --------------------------------------------------
# V3 training configuration
# --------------------------------------------------

@export var training_episodes := 5000
@export var alpha := 0.015
@export var gamma := 0.98
@export var epsilon_start := 1.0
@export var epsilon_end := 0.05
@export var epsilon_decay := 0.9995

@export var save_weights_after_training := true
@export var load_trained_weights := false

# Accelerated visual training. Training continues every episode; the visible
# map is refreshed only at this interval.
@export var visual_training_interval := 25

# --------------------------------------------------
# Persistence
# --------------------------------------------------

const WEIGHTS_PATH := "user://weights_v3.json"
const TRAINING_LOG_PATH := "user://training_log_v3.json"

# --------------------------------------------------
# Playback
# --------------------------------------------------

@export var playback_step_delay := 0.20
@export var max_playback_steps := 1152

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
var training_in_progress := false


func _ready() -> void:
	grid_world = GridWorld.new()
	grid_renderer = GridRenderer.new()
	q_learning = QLearning.new(12345)
	trainer = Trainer.new(q_learning, agent, grid_world)

	add_child(grid_world)
	add_child(grid_renderer)

	setup_demo_map(MAP_SEED)

	if hud != null:
		hud.train_requested.connect(run_configured_training)
		hud.trained_playback_requested.connect(start_trained_playback)
		hud.random_playback_requested.connect(start_random_playback)
		hud.new_map_requested.connect(randomize_current_map)

		hud.set_map_seed(MAP_SEED)
		# There is no Q-table in V3. Keep the existing HUD field populated with
		# the number of learned model parameters so the UI remains intact.
		hud.set_states_learned(q_learning.weights.size())

	print("")
	print("==========================================")
	print("RL GRID WORLD - V3 LINEAR Q LEARNING")
	print("==========================================")
	print("Current map seed: ", MAP_SEED)
	print("Weights path: ", WEIGHTS_PATH)
	print("Feature count: ", QLearning.FEATURE_COUNT)
	print("==========================================")
	print("")

	if load_trained_weights:
		load_configured_weights()


# --------------------------------------------------
# Map setup / reset
# --------------------------------------------------

func setup_demo_map(map_seed: int) -> void:
	playback_running = false
	playback_timer = 0.0
	playback_steps = 0
	playback_total_reward = 0.0

	var grid := grid_world.generate_map(map_seed)
	grid_renderer.render_grid(grid, grid_map)

	var start_position := grid_world.find_agent_start(grid)
	agent.setup(start_position, grid, grid_renderer, gamma)

	playback_start_collectibles = agent.get_collectible_count()

	if hud != null:
		hud.set_map_seed(map_seed)
		hud.set_states_learned(q_learning.weights.size())

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
	if training_in_progress:
		print("")
		print("Cannot randomize map while training is in progress.")
		print("")
		return

	playback_running = false
	playback_timer = 0.0

	var old_seed := MAP_SEED
	MAP_SEED = randi_range(1, 2147483647)
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

	setup_demo_map(MAP_SEED)

	print("✓ New map generated (Seed %d)" % MAP_SEED)


# --------------------------------------------------
# Weight loading
# --------------------------------------------------

func load_configured_weights() -> bool:
	print("")
	print("Loading V3 trained weights")
	print("Path: ", WEIGHTS_PATH)

	var loaded := q_learning.load_weights(WEIGHTS_PATH)
	print("Weight load successful: ", loaded)

	if loaded and hud != null:
		hud.set_states_learned(q_learning.weights.size())

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

	if hud != null:
		hud.on_training_started(training_episodes)

	print("")
	print("==========================================")
	print("STARTING V3 TRAINING")
	print("==========================================")
	print("Episodes: ", training_episodes)
	print("Alpha: ", alpha)
	print("Gamma: ", gamma)
	print("Epsilon start: ", epsilon_start)
	print("Epsilon min: ", epsilon_end)
	print("Epsilon decay: ", epsilon_decay)
	print("Visual refresh: every ", visual_training_interval, " episodes")
	print("")

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

	var log_saved := trainer.save_training_log(TRAINING_LOG_PATH)
	var weights_saved := false

	if save_weights_after_training:
		weights_saved = q_learning.save_weights(WEIGHTS_PATH)

	print("")
	print("==========================================")
	print("TRAINING COMPLETE")
	print("==========================================")
	print("Episodes completed: ", rewards.size())
	print("Learned weights: ", q_learning.weights)
	print("Training log save successful: ", log_saved)
	print("Weights save successful: ", weights_saved)
	print("")

	if hud != null:
		hud.on_training_evaluating()

	# V3 evaluates against separately seeded, unseen random maps.
	var evaluation_results := trainer.evaluate_test_maps()

	if hud != null:
		hud.on_evaluation_finished(evaluation_results)
		hud.on_training_finished({
			"total_episodes": rewards.size(),
			"states_learned": q_learning.weights.size(),
		})

	print("LEARNED AGENT UNSEEN-MAP EVALUATION")
	for result in evaluation_results:
		print(
			"Seed: ", result["seed"],
			" | Reward: ", result["reward"],
			" | Steps: ", result["steps"],
			" | Completed: ", result["completed"],
			" | Remaining collectibles: ", result["remaining_collectibles"],
		)

	print("")

	training_in_progress = false
	setup_demo_map(MAP_SEED)

	print("Training finished.")
	print("")


func _on_training_snapshot(data: Dictionary) -> void:
	var snapshot_grid: Array = data.get("grid_snapshot", agent.grid_data)
	var snapshot_position: Vector2i = data.get("position", agent.grid_position)

	grid_renderer.render_grid(snapshot_grid, grid_map)

	var snapshot_world_position := grid_renderer.grid_to_world(snapshot_position)
	agent.position = snapshot_world_position
	agent.visual_target_position = snapshot_world_position

	if hud != null:
		data["states_count"] = q_learning.weights.size()
		hud.on_training_progress(data)

	print(
		"Training snapshot | Episode ", data["episode"],
		"/", data["total_episodes"],
		" | Seed: ", data["seed"],
		" | Reward: ", data["reward"],
		" | Epsilon: ", data["epsilon"],
		" | Items: ", data.get("items", 0),
		"/", trainer.NUM_COLLECTIBLES,
		" | Success: ", data.get("success", false),
		" | Start position: ", data["position"],
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

	if not load_configured_weights():
		print("ERROR: Could not load trained V3 weights. Train first!")
		playback_running = false
		return

	setup_demo_map(MAP_SEED)

	playback_mode = PlaybackMode.TRAINED
	playback_running = true
	playback_timer = playback_step_delay

	if hud != null:
		hud.on_playback_started("Trained", playback_start_collectibles)

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

	setup_demo_map(MAP_SEED)

	playback_mode = PlaybackMode.RANDOM
	playback_running = true
	playback_timer = playback_step_delay

	if hud != null:
		hud.on_playback_started("Random", playback_start_collectibles)

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

	if playback_steps >= max_playback_steps:
		finish_playback(false)
		return

	var action: int
	var q_summary := ""

	if playback_mode == PlaybackMode.TRAINED:
		var phi := agent.get_features()
		var q_values := q_learning.q_values(phi)
		action = q_learning.get_action(phi, 0.0)
		q_summary = "Q=%s" % str(q_values)

	elif playback_mode == PlaybackMode.RANDOM:
		action = randi_range(0, QLearning.ACTION_COUNT - 1)

	else:
		return

	var result: Dictionary = agent.try_move(action, gamma)
	var reward := float(result["reward"])

	playback_total_reward += reward
	playback_steps += 1

	if result["collected"] or result["completed"]:
		grid_renderer.render_grid(agent.grid_data, grid_map)

	if hud != null:
		hud.on_playback_step({
			"step": playback_steps,
			"total_reward": playback_total_reward,
			"remaining": agent.get_collectible_count(),
		})

	print(
		"Playback step ", playback_steps,
		" | Action: ", action_name(action),
		" | Reward: ", reward,
		" | Total reward: ", playback_total_reward,
		" | Remaining collectibles: ", agent.get_collectible_count(),
		" | ", q_summary,
	)

	if result["completed"] or result["done"]:
		finish_playback(bool(result["completed"]))


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
		"/", playback_start_collectibles,
	)
	print("Remaining collectibles: ", agent.get_collectible_count())
	print("Completed: ", completed)
	print("==========================================")
	print("")

	if hud != null:
		hud.on_playback_finished({
			"policy_name": policy_name,
			"completed": completed,
			"steps": playback_steps,
			"reward": playback_total_reward,
		})


# --------------------------------------------------
# Utility
# --------------------------------------------------

func action_name(action: int) -> String:
	match action:
		0:
			return "UP"
		1:
			return "DOWN"
		2:
			return "LEFT"
		3:
			return "RIGHT"

	return "UNKNOWN"
