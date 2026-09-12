extends Node3D
class_name GridAgent


enum Action {
	UP,
	DOWN,
	LEFT,
	RIGHT
}


const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3


const DIRECTIONS := {
	Action.UP: Vector2i(0, -1),
	Action.DOWN: Vector2i(0, 1),
	Action.LEFT: Vector2i(-1, 0),
	Action.RIGHT: Vector2i(1, 0)
}


var grid_position := Vector2i.ZERO
var grid_data: Array = []
var grid_renderer: GridRenderer


func setup(
	start_position: Vector2i,
	world_grid: Array,
	renderer: GridRenderer
) -> void:

	grid_position = start_position
	grid_data = world_grid
	grid_renderer = renderer

	update_visual_position()


func try_move(action: int) -> Dictionary:
	if not DIRECTIONS.has(action):
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"position": grid_position
		}

	var direction: Vector2i = DIRECTIONS[action]

	var target_position := grid_position + direction

	# Check whether the target is inside the grid.
	if not is_inside_grid(target_position):
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"position": grid_position
		}

	# Check whether the target is an obstacle.
	if grid_data[target_position.y][target_position.x] == OBSTACLE:
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"position": grid_position
		}

	# Move the agent.
	grid_position = target_position

	var collected := false

	# Check whether the agent entered a collectible cell.
	if grid_data[grid_position.y][grid_position.x] == COLLECTIBLE:
		grid_data[grid_position.y][grid_position.x] = EMPTY
		collected = true

	update_visual_position()

	return {
		"success": true,
		"blocked": false,
		"collected": collected,
		"position": grid_position
	}


func is_inside_grid(position: Vector2i) -> bool:
	if position.y < 0:
		return false

	if position.y >= grid_data.size():
		return false

	if position.x < 0:
		return false

	if position.x >= grid_data[position.y].size():
		return false

	return true


func update_visual_position() -> void:
	if grid_renderer == null:
		return

	position = grid_renderer.grid_to_world(
		grid_position
	)
