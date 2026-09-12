extends Node3D

@onready var grid_map: GridMap = $GridMap
@onready var agent: GridAgent = $Agent
@onready var camera: Camera3D = $Camera3D

@export var MAP_SEED := 42

var grid_world: GridWorld
var grid_renderer: GridRenderer

func position_camera(grid_size: int) -> void:
	var center := Vector3(grid_size / 2.0, 0, grid_size / 2.0)
	var horizontal_distance := grid_size * 0.9
	var height := grid_size * 1.6  # taller relative to horizontal = steeper, more top-down

	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = grid_size * 1.15
	camera.position = center + Vector3(horizontal_distance, height, horizontal_distance)
	camera.look_at(center, Vector3.UP)

func _ready() -> void:
	grid_world = GridWorld.new()
	grid_renderer = GridRenderer.new()

	add_child(grid_world)
	add_child(grid_renderer)

	var grid := grid_world.generate_map(MAP_SEED)

	grid_renderer.render_grid(grid, grid_map)

	var start_position := find_agent_start(grid)

	agent.setup(
		start_position,
		grid,
		grid_renderer
	)

	position_camera(GridWorld.GRID_SIZE)

	print("Generated map with seed ", MAP_SEED)
	grid_world.print_map(grid)


func find_agent_start(grid: Array) -> Vector2i:
	for y in range(grid.size()):
		for x in range(grid[y].size()):
			if grid[y][x] == GridWorld.AGENT_START:
				return Vector2i(x, y)

	return Vector2i.ZERO

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:

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
		" | Position: ",
		result.position,
		" | Blocked: ",
		result.blocked,
		" | Collected: ",
		result.collected
	)
