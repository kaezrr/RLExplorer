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


# --------------------------------------------------
# Goal-direction encoding.
#
# This is now the first *action* to take along the actual
# (obstacle-aware) shortest path to the nearest reachable
# collectible, computed via BFS - not a straight-line compass
# bucket. It shares the same integer values as Action so that
# "go UP" as a goal hint and Action.UP line up, plus one extra
# value for "no path / nothing to collect".
# --------------------------------------------------

const GOAL_DIR_NONE := 4


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

# Last action taken (or attempted, even if blocked).
# -1 means "no previous action" (start of episode).
#
# This is folded into the state key so that arriving at a cell
# fresh and arriving at it having just reversed out of it are
# distinguishable states. Without this, two adjacent cells whose
# greedy actions point at each other create an infinite,
# undetectable back-and-forth loop once epsilon reaches 0.
var last_action: int = -1


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

	# Reset per-episode memory.
	last_action = -1

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

	# Record the attempted action before resolving it, so the next
	# state key reflects "just tried/took this direction" even if
	# the move turns out to be blocked.
	last_action = action

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

	var goal_direction := get_goal_direction()

	# Shift last_action (-1..3) to a non-negative range (0..4)
	# so it can sit cleanly in the state string.
	var prev_action_component := last_action + 1


	return "%d,%d,%d,%d,%d,%d" % [
		up_type,
		down_type,
		left_type,
		right_type,
		goal_direction,
		prev_action_component
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


# --------------------------------------------------
# Obstacle-aware goal direction.
#
# Runs a breadth-first search out from the agent's current
# position, respecting obstacles and grid bounds exactly like
# Reachability.get_reachable_cells(). The first frontier cell
# that is a collectible tells us the nearest *reachable*
# collectible by actual path length (not straight-line
# distance), and the direction of the very first step taken to
# reach it (tracked alongside each queued cell) becomes the
# goal-direction feature.
#
# This replaces the old Manhattan-distance compass bucket, which
# could point straight through a wall whenever the nearest
# collectible in a straight line wasn't the nearest one by an
# actual walkable path.
# --------------------------------------------------

func get_goal_direction() -> int:

	var visited := {}
	var queue: Array = []

	visited[grid_position] = true
	queue.append({
		"pos": grid_position,
		"first_action": -1
	})

	while not queue.is_empty():
		var current: Dictionary = queue.pop_front()
		var current_pos: Vector2i = current["pos"]
		var first_action: int = current["first_action"]

		if grid_data[current_pos.y][current_pos.x] == COLLECTIBLE:
			# The agent's own starting cell is never a collectible
			# mid-episode (it would already have been collected),
			# so first_action is guaranteed to be a real action here.
			return first_action

		for action in DIRECTIONS:
			var direction: Vector2i = DIRECTIONS[action]
			var next_pos: Vector2i = current_pos + direction

			if not is_inside_grid(next_pos):
				continue

			if grid_data[next_pos.y][next_pos.x] == OBSTACLE:
				continue

			if visited.has(next_pos):
				continue

			visited[next_pos] = true

			var next_first_action: int = first_action

			if next_first_action == -1:
				next_first_action = action

			queue.append({
				"pos": next_pos,
				"first_action": next_first_action
			})

	# No reachable collectible left (shouldn't normally happen
	# mid-episode on a validated map, but covered defensively).
	return GOAL_DIR_NONE


func get_collectible_count() -> int:

	var count := 0


	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):

			if grid_data[y][x] == COLLECTIBLE:
				count += 1


	return count
