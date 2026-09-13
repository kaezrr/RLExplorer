extends Node3D

@onready var grid_map: GridMap = $GridMap
@onready var agent: GridAgent = $Agent
@onready var camera: Camera3D = $Camera3D

@export var MAP_SEED := 42

# Training controls.
@export var run_training := false
@export var training_episodes := 5000
@export var alpha := 0.1
@export var gamma := 0.9
@export var epsilon_start := 1.0
@export var epsilon_end := 0.05
@export var epsilon_decay := 0.995

var grid_world: GridWorld
var grid_renderer: GridRenderer
var q_learning: QLearning


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


func _ready() -> void:
	grid_world = GridWorld.new()
	grid_renderer = GridRenderer.new()
	q_learning = QLearning.new()

	add_child(grid_world)
	add_child(grid_renderer)

	var grid := grid_world.generate_map(MAP_SEED)
	grid_renderer.render_grid(grid, grid_map)

	var start_position := find_agent_start(grid)
	agent.setup(start_position, grid, grid_renderer)

	print("Initial agent state: ", agent.get_state_key())

	position_camera(GridWorld.GRID_SIZE)

	if run_training:
		run_configured_training()


func run_configured_training() -> void:
	var trainer := Trainer.new(
		q_learning,
		agent,
		grid_world
	)

	print("========================================")
	print("Starting configured training")
	print("Episodes: ", training_episodes)
	print("Training seeds: ", trainer.training_seeds)
	print("Alpha: ", alpha)
	print("Gamma: ", gamma)
	print("Epsilon start: ", epsilon_start)
	print("Epsilon end: ", epsilon_end)
	print("Epsilon decay: ", epsilon_decay)
	print("========================================")

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

	print("========================================")
	print("Training complete")
	print("Episodes completed: ", rewards.size())
	print("Q-table states learned: ", q_learning.q_table.size())
	print("Training log save successful: ", log_saved)
	print("========================================")


func find_agent_start(grid: Array) -> Vector2i:
	for y in range(grid.size()):
		for x in range(grid[y].size()):
			if grid[y][x] == GridWorld.AGENT_START:
				return Vector2i(x, y)

	return Vector2i.ZERO


func _unhandled_input(event: InputEvent) -> void:
	if not event is InputEventKey:
		return

	if not event.pressed or event.echo:
		return

	match event.keycode:
		KEY_W:
			test_agent_move(GridAgent.Action.UP)
		KEY_S:
			test_agent_move(GridAgent.Action.DOWN)
		KEY_A:
			test_agent_move(GridAgent.Action.LEFT)
		KEY_D:
			test_agent_move(GridAgent.Action.RIGHT)


func test_agent_move(action: int) -> void:
	var result := agent.try_move(action)

	print(
		"Action: ",
		action,
		" | Result: ",
		result,
		" | State: ",
		agent.get_state_key()
	)
