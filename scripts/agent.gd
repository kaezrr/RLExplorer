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


const DIRECTION_N := 0
const DIRECTION_NE := 1
const DIRECTION_E := 2
const DIRECTION_SE := 3
const DIRECTION_S := 4
const DIRECTION_SW := 5
const DIRECTION_W := 6
const DIRECTION_NW := 7


const STEP_REWARD := -0.1
const BLOCKED_REWARD := -2.0
const COLLECTIBLE_REWARD := 10.0
const ALL_COLLECTIBLES_BONUS := 50.0


# --------------------------------------------------
# Visual movement settings.
# --------------------------------------------------

@export var smooth_movement := true
@export var movement_speed := 5.0

var visual_target_position := Vector3.ZERO
var visual_position_initialized := false


# --------------------------------------------------
# Agent state.
# --------------------------------------------------

var grid_position := Vector2i.ZERO
var grid_data: Array = []
var grid_renderer: GridRenderer


func _ready() -> void:
	set_process(true)


func _process(delta: float) -> void:
	if not smooth_movement:
		return

	if not visual_position_initialized:
		return

	# Move the visual agent smoothly toward the
	# logical grid position.
	position = position.move_toward(
		visual_target_position,
		movement_speed * delta
	)


func setup(
	start_position: Vector2i,
	world_grid: Array,
	renderer: GridRenderer
) -> void:

	grid_position = start_position
	grid_data = world_grid
	grid_renderer = renderer

	# Reset the visual position immediately when
	# starting a new map/episode.
	var start_world_position := grid_renderer.grid_to_world(
		grid_position
	)

	position = start_world_position
	visual_target_position = start_world_position
	visual_position_initialized = true


func try_move(action: int) -> Dictionary:

	# --------------------------------------------------
	# Invalid action.
	# --------------------------------------------------

	if not DIRECTIONS.has(action):
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"completed": false,
			"reward": BLOCKED_REWARD,
			"position": grid_position
		}


	var direction: Vector2i = DIRECTIONS[action]

	var target_position := grid_position + direction


	# --------------------------------------------------
	# Off-grid movement.
	# --------------------------------------------------

	if not is_inside_grid(target_position):
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"completed": false,
			"reward": BLOCKED_REWARD,
			"position": grid_position
		}


	# --------------------------------------------------
	# Obstacle collision.
	# --------------------------------------------------

	if grid_data[target_position.y][target_position.x] == OBSTACLE:
		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"completed": false,
			"reward": BLOCKED_REWARD,
			"position": grid_position
		}


	# --------------------------------------------------
	# Logical movement.
	#
	# IMPORTANT:
	# The RL environment still changes grid_position
	# immediately. Only the visual representation moves
	# smoothly.
	# --------------------------------------------------

	grid_position = target_position


	var collected := false


	if grid_data[grid_position.y][grid_position.x] == COLLECTIBLE:
		grid_data[grid_position.y][grid_position.x] = EMPTY
		collected = true


	# --------------------------------------------------
	# Check whether every collectible has been collected.
	# --------------------------------------------------

	var completed := get_collectible_count() == 0


	var reward := STEP_REWARD


	if collected:
		reward += COLLECTIBLE_REWARD


	if completed:
		reward += ALL_COLLECTIBLES_BONUS


	# --------------------------------------------------
	# Update the visual target.
	#
	# The agent does NOT snap to the new position here.
	# _process() moves it smoothly.
	# --------------------------------------------------

	update_visual_position()


	return {
		"success": true,
		"blocked": false,
		"collected": collected,
		"completed": completed,
		"reward": reward,
		"position": grid_position
	}


func is_inside_grid(world_position: Vector2i) -> bool:

	if world_position.y < 0:
		return false

	if world_position.y >= grid_data.size():
		return false

	if world_position.x < 0:
		return false

	if world_position.x >= grid_data[world_position.y].size():
		return false

	return true


func update_visual_position() -> void:

	if grid_renderer == null:
		return

	visual_target_position = grid_renderer.grid_to_world(
		grid_position
	)

	# If smoothing is disabled, immediately place the agent.
	if not smooth_movement:
		position = visual_target_position


func get_state_key() -> String:

	var up_type := get_adjacent_cell_type(
		grid_position + Vector2i.UP
	)

	var down_type := get_adjacent_cell_type(
		grid_position + Vector2i.DOWN
	)

	var left_type := get_adjacent_cell_type(
		grid_position + Vector2i.LEFT
	)

	var right_type := get_adjacent_cell_type(
		grid_position + Vector2i.RIGHT
	)

	var goal_direction := get_nearest_collectible_direction()


	return "%d,%d,%d,%d,%d" % [
		up_type,
		down_type,
		left_type,
		right_type,
		goal_direction
	]


func get_adjacent_cell_type(
	position_to_check: Vector2i
) -> int:

	# Outside the grid is treated exactly like an obstacle.
	if not is_inside_grid(position_to_check):
		return 1

	# Only obstacles are represented as blocked.
	if grid_data[position_to_check.y][position_to_check.x] == OBSTACLE:
		return 1

	return 0


func get_nearest_collectible_direction() -> int:

	var nearest_position := Vector2i.ZERO
	var nearest_distance := INF
	var found_collectible := false


	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):

			if grid_data[y][x] != COLLECTIBLE:
				continue

			var collectible_position := Vector2i(x, y)


			var distance: float = (
				abs(collectible_position.x - grid_position.x)
				+ abs(collectible_position.y - grid_position.y)
			)


			if distance < nearest_distance:
				nearest_distance = distance
				nearest_position = collectible_position
				found_collectible = true


	if not found_collectible:
		# This case should normally only occur at episode completion.
		return DIRECTION_N


	var dx := nearest_position.x - grid_position.x
	var dy := nearest_position.y - grid_position.y


	return get_direction_bucket(dx, dy)


func get_direction_bucket(dx: int, dy: int) -> int:

	var horizontal: int = sign(dx)
	var vertical: int = sign(dy)


	# Godot grid convention:
	# y decreases when moving UP.

	if horizontal == 0 and vertical < 0:
		return DIRECTION_N

	if horizontal > 0 and vertical < 0:
		return DIRECTION_NE

	if horizontal > 0 and vertical == 0:
		return DIRECTION_E

	if horizontal > 0 and vertical > 0:
		return DIRECTION_SE

	if horizontal == 0 and vertical > 0:
		return DIRECTION_S

	if horizontal < 0 and vertical > 0:
		return DIRECTION_SW

	if horizontal < 0 and vertical == 0:
		return DIRECTION_W

	if horizontal < 0 and vertical < 0:
		return DIRECTION_NW


	# No direction if dx == 0 and dy == 0.
	return DIRECTION_N


func get_collectible_count() -> int:

	var count := 0


	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):

			if grid_data[y][x] == COLLECTIBLE:
				count += 1


	return count
