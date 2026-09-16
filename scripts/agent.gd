extends Node3D
class_name GridAgent

## V3 GridWorld agent/environment.
##
## The learned policy does NOT consume a discrete state key. Instead, this
## environment builds a [4][10] feature matrix, one 10-D feature vector per
## action. The same ten weights are reused for every cell and every map.
##
## Feature order (must match test_rl_3.py exactly):
##   0 wall_a       : neighbour is blocked
##   1 item_a       : neighbour contains a remaining collectible
##   2 visited_a    : neighbour has already been visited (blocked => 1)
##   3 dist_d_a     : normalized Manhattan-distance decrease to nearest item
##   4 align_a      : alignment with nearest-item beacon
##   5 deadend_a    : neighbour is a degree-1 dead-end without an item
##   6 bias         : constant 1
##   7 back_a       : neighbour is the cell we just came from
##   8 frontier_a   : unvisited neighbour with >=2 open exits
##   9 align_c_a    : alignment with centroid of all remaining items

const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3

const ACTION_COUNT := 4
const FEAT_DIM := 10

const GOAL_DIR_NONE := 4 # Kept for compatibility with older UI/debug code.

# V3 environment/reward configuration.
const GRID_W := 12
const GRID_H := 12
const OBSTACLE_DENSITY := 0.18
const NUM_COLLECTIBLES := 6
const MAX_STEPS_MULTIPLIER := 8.0
const MAX_STEPS := int(GRID_W * GRID_H * MAX_STEPS_MULTIPLIER)

const STEP_REWARD := -0.04
const WALL_PENALTY := -1.0
const COLLECT_REWARD := 12.0
const FINAL_COLLECT_REWARD := 30.0
const TIMEOUT_PENALTY := -6.0
const BACKTRACK_PENALTY := -0.25
const NOVELTY_REWARD := 0.20
const SHAPING_SCALE := 2.0
const GAMMA_DEFAULT := 0.98

# 0=UP 1=DOWN 2=LEFT 3=RIGHT.
# Godot uses x=column, y=row, so the first component maps to row delta
# and the second component maps to column delta in the Python reference.
const DIRECTIONS := {
	0: Vector2i(0, -1),
	1: Vector2i(0, 1),
	2: Vector2i(-1, 0),
	3: Vector2i(1, 0),
}

@export var smooth_movement := true
@export var movement_speed := 5.0

var visual_target_position := Vector3.ZERO
var visual_position_initialized := false

# Episode state.
var grid_position := Vector2i.ZERO
var grid_data: Array = []
var grid_renderer: GridRenderer
var start_position := Vector2i.ZERO

var remaining_collectibles: Array[Vector2i] = []
var visit_counts: Array = []
var trajectory: Array[Vector2i] = []
var previous_position: Variant = null

var steps := 0
var total_reward := 0.0
var new_cells := 0
var repeated_visits := 0
var wall_hits := 0
var backtracks := 0

var _prev_potential := 0.0


func _ready() -> void:
	set_process(true)


func _process(delta: float) -> void:
	if not smooth_movement or not visual_position_initialized:
		return

	position = position.move_toward(
		visual_target_position,
		movement_speed * delta
	)


func setup(
	new_start_position: Vector2i,
	world_grid: Array,
	renderer: GridRenderer,
	gamma: float = GAMMA_DEFAULT,
) -> void:
	grid_position = new_start_position
	start_position = new_start_position
	grid_data = _duplicate_grid(world_grid)
	grid_renderer = renderer

	remaining_collectibles.clear()
	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):
			if grid_data[y][x] == COLLECTIBLE:
				remaining_collectibles.append(Vector2i(x, y))

	visit_counts.clear()
	for y in range(grid_data.size()):
		var row: Array[int] = []
		row.resize(grid_data[y].size())
		row.fill(0)
		visit_counts.append(row)

	visit_counts[grid_position.y][grid_position.x] = 1
	trajectory.clear()
	trajectory.append(grid_position)
	previous_position = null

	steps = 0
	total_reward = 0.0
	new_cells = 1
	repeated_visits = 0
	wall_hits = 0
	backtracks = 0
	_prev_potential = _potential(gamma)

	var start_world_position := _grid_to_world(grid_position)
	position = start_world_position
	visual_target_position = start_world_position
	visual_position_initialized = true


func try_move(action: int, gamma: float = GAMMA_DEFAULT) -> Dictionary:
	steps += 1

	# Match the reference environment defensively for invalid actions.
	if action < 0 or action >= ACTION_COUNT:
		var invalid_reward := STEP_REWARD + WALL_PENALTY
		wall_hits += 1
		var invalid_done := steps >= MAX_STEPS
		if invalid_done:
			invalid_reward += TIMEOUT_PENALTY
			total_reward += invalid_reward
			trajectory.append(grid_position)
			previous_position = grid_position
		else:
			total_reward += invalid_reward
			trajectory.append(grid_position)
			previous_position = grid_position

		return {
			"success": false,
			"blocked": true,
			"collected": false,
			"completed": false,
			"done": invalid_done,
			"reward": invalid_reward,
			"position": grid_position,
			"items_collected": get_items_collected(),
		}

	var direction: Vector2i = DIRECTIONS[action]
	var old_position := grid_position
	var target_position := grid_position + direction

	var reward := STEP_REWARD
	var done := false
	var collected := false
	var new_cell := false
	var backtracked := false
	var blocked := false

	# --------------------------------------------------
	# Movement / collision.
	# --------------------------------------------------
	if not is_inside_grid(target_position) or is_obstacle(target_position):
		blocked = true
		reward += WALL_PENALTY
		wall_hits += 1
		# Stay in place, matching the Python environment.
	else:
		if previous_position != null and target_position == previous_position:
			backtracked = true
			backtracks += 1
			reward += BACKTRACK_PENALTY

		grid_position = target_position

		visit_counts[grid_position.y][grid_position.x] += 1
		if visit_counts[grid_position.y][grid_position.x] == 1:
			new_cell = true
			new_cells += 1
		else:
			repeated_visits += 1

	# --------------------------------------------------
	# Collectible pickup.
	# --------------------------------------------------
	if not blocked and is_collectible(grid_position):
		collected = true
		_remove_collectible(grid_position)
		grid_data[grid_position.y][grid_position.x] = EMPTY
		reward += COLLECT_REWARD

		if remaining_collectibles.is_empty():
			reward += FINAL_COLLECT_REWARD
			done = true

	# --------------------------------------------------
	# Intrinsic novelty.
	# --------------------------------------------------
	if new_cell:
		reward += NOVELTY_REWARD

	# --------------------------------------------------
	# Potential-based shaping.
	# --------------------------------------------------
	var new_potential := _potential(gamma)
	reward += SHAPING_SCALE * (gamma * new_potential - _prev_potential)
	_prev_potential = new_potential

	# --------------------------------------------------
	# Timeout.
	# --------------------------------------------------
	if steps >= MAX_STEPS and not done:
		reward += TIMEOUT_PENALTY
		done = true

	# Maintain transition history exactly enough for the V3 backtrack feature.
	trajectory.append(grid_position)
	previous_position = old_position
	total_reward += reward

	update_visual_position()

	return {
		"success": not blocked,
		"blocked": blocked,
		"collected": collected,
		"completed": remaining_collectibles.is_empty(),
		"done": done,
		"reward": reward,
		"position": grid_position,
		"new_cell": new_cell,
		"backtracked": backtracked,
		"items_collected": get_items_collected(),
		"coverage": get_coverage(),
	}


## Returns the V3 observation: 4 actions x 10 features.
func get_features() -> Array:
	var features: Array = []
	for action in range(ACTION_COUNT):
		features.append(_compute_action_features(action))
	return features


## Alias for callers that prefer the terminology used in the Python file.
func compute_features() -> Array:
	return get_features()


func _compute_action_features(action: int) -> Array[float]:
	var phi: Array[float] = []
	phi.resize(FEAT_DIM)
	phi.fill(0.0)

	if action < 0 or action >= ACTION_COUNT:
		return phi

	var target: Vector2i = grid_position + DIRECTIONS[action]
	var blocked := not is_inside_grid(target) or is_obstacle(target)

	# bias is present for every action, including blocked actions.
	phi[6] = 1.0

	if blocked:
		phi[0] = 1.0
		phi[2] = 1.0
		return phi

	phi[1] = 1.0 if is_collectible(target) else 0.0
	phi[2] = 1.0 if _visit_count(target) > 0 else 0.0

	if not remaining_collectibles.is_empty():
		var nearest := _nearest_collectible()
		var cur_d :int= abs(nearest.y - grid_position.y) + abs(nearest.x - grid_position.x)
		var new_d :int= abs(nearest.y - target.y) + abs(nearest.x - target.x)
		phi[3] = float(cur_d - new_d) / float(max(1, grid_data.size() + _grid_width()))

		var beacon := _unit_beacon_to(grid_position, nearest)
		var dir_unit := _direction_unit(action)
		phi[4] = dir_unit.dot(beacon)
		phi[9] = _centroid_alignment(action)

	var degree := _cell_degree(target)
	if degree <= 1 and not is_collectible(target):
		phi[5] = 1.0

	if previous_position != null and target == previous_position:
		phi[7] = 1.0

	if _visit_count(target) == 0 and degree >= 2:
		phi[8] = 1.0

	return phi


func _direction_unit(action: int) -> Vector2:
	var d: Vector2i = DIRECTIONS[action]
	# Feature-space convention is [row, col], but a 2-D vector is sufficient
	# here because the dot product uses the same ordering for both terms.
	return Vector2(float(d.y), float(d.x))


func _nearest_collectible() -> Vector2i:
	var best := remaining_collectibles[0]
	var best_d :int= abs(best.y - grid_position.y) + abs(best.x - grid_position.x)

	for item in remaining_collectibles:
		var d :int= abs(item.y - grid_position.y) + abs(item.x - grid_position.x)
		if d < best_d:
			best = item
			best_d = d

	return best


func _unit_beacon_to(from: Vector2i, to: Vector2i) -> Vector2:
	var row_delta := float(to.y - from.y)
	var col_delta := float(to.x - from.x)
	var length := sqrt(row_delta * row_delta + col_delta * col_delta)
	if length <= 0.000001:
		return Vector2.ZERO
	return Vector2(row_delta / length, col_delta / length)


func _centroid_alignment(action: int) -> float:
	if remaining_collectibles.is_empty():
		return 0.0

	var row_sum := 0.0
	var col_sum := 0.0
	for item in remaining_collectibles:
		row_sum += float(item.y)
		col_sum += float(item.x)

	var centroid_row := row_sum / float(remaining_collectibles.size())
	var centroid_col := col_sum / float(remaining_collectibles.size())
	var row_delta := centroid_row - float(grid_position.y)
	var col_delta := centroid_col - float(grid_position.x)
	var length := sqrt(row_delta * row_delta + col_delta * col_delta)
	if length <= 0.000001:
		return 0.0

	var beacon := Vector2(row_delta / length, col_delta / length)
	return _direction_unit(action).dot(beacon)


func _cell_degree(cell: Vector2i) -> int:
	var degree := 0
	for action in range(ACTION_COUNT):
		var neighbour :Vector2i= cell + DIRECTIONS[action]
		if is_inside_grid(neighbour) and not is_obstacle(neighbour):
			degree += 1
	return degree


func is_inside_grid(cell: Vector2i) -> bool:
	if cell.y < 0 or cell.y >= grid_data.size():
		return false
	if cell.x < 0 or cell.x >= grid_data[cell.y].size():
		return false
	return true


func is_obstacle(cell: Vector2i) -> bool:
	return is_inside_grid(cell) and grid_data[cell.y][cell.x] == OBSTACLE


func is_collectible(cell: Vector2i) -> bool:
	return _contains_collectible(cell)


func _contains_collectible(cell: Vector2i) -> bool:
	for item in remaining_collectibles:
		if item == cell:
			return true
	return false


func _remove_collectible(cell: Vector2i) -> void:
	for i in range(remaining_collectibles.size()):
		if remaining_collectibles[i] == cell:
			remaining_collectibles.remove_at(i)
			return


func _visit_count(cell: Vector2i) -> int:
	if cell.y < 0 or cell.y >= visit_counts.size():
		return 0
	if cell.x < 0 or cell.x >= visit_counts[cell.y].size():
		return 0
	return int(visit_counts[cell.y][cell.x])


func get_collectible_count() -> int:
	return remaining_collectibles.size()


func get_items_collected() -> int:
	var total := 0
	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):
			if grid_data[y][x] == EMPTY:
				# Don't infer collectibles from the mutated grid because normal empty
				# cells are indistinguishable. Use original count from remaining count.
				pass

	# V3 maps always begin with NUM_COLLECTIBLES items.
	return NUM_COLLECTIBLES - remaining_collectibles.size()


func get_coverage() -> float:
	var walkable := 0
	for row in grid_data:
		for cell in row:
			if cell != OBSTACLE:
				walkable += 1
	return float(new_cells) / float(max(1, walkable))


func get_goal_direction() -> int:
	# Compatibility helper only. This is intentionally NOT used for policy
	# decisions. It returns the direction of the closest item by Manhattan
	# geometry, matching the beacon's immediate sign when possible.
	if remaining_collectibles.is_empty():
		return GOAL_DIR_NONE

	var item := _nearest_collectible()
	var dr := item.y - grid_position.y
	var dc := item.x - grid_position.x

	if abs(dr) >= abs(dc) and dr != 0:
		return 0 if dr < 0 else 1
	if dc != 0:
		return 2 if dc < 0 else 3
	return GOAL_DIR_NONE


func update_visual_position() -> void:
	if grid_renderer == null:
		return

	visual_target_position = grid_renderer.grid_to_world(grid_position)
	if not smooth_movement:
		position = visual_target_position


func _grid_to_world(cell: Vector2i) -> Vector3:
	if grid_renderer != null:
		return grid_renderer.grid_to_world(cell)
	return Vector3(float(cell.x), 0.0, float(cell.y))


func _duplicate_grid(source: Array) -> Array:
	var copy: Array = []
	for row in source:
		copy.append(row.duplicate())
	return copy


func _grid_width() -> int:
	if grid_data.is_empty():
		return 0
	return grid_data[0].size()


## Exact V3 potential: negative BFS distance to nearest remaining item.
func _potential(_gamma: float = GAMMA_DEFAULT) -> float:
	if remaining_collectibles.is_empty():
		return 0.0

	var distance := _bfs_nearest_item_distance()
	if distance < 0:
		return 0.0
	return -float(distance)


func _bfs_nearest_item_distance() -> int:
	if remaining_collectibles.is_empty():
		return 0

	var queue: Array[Vector2i] = []
	var distances: Dictionary = {}
	queue.append(grid_position)
	distances[grid_position] = 0

	var head := 0
	while head < queue.size():
		var current := queue[head]
		head += 1
		var d := int(distances[current])

		if _contains_collectible(current):
			return d

		for action in range(ACTION_COUNT):
			var next :Vector2i= current + DIRECTIONS[action]
			if not is_inside_grid(next) or is_obstacle(next):
				continue
			if distances.has(next):
				continue
			distances[next] = d + 1
			queue.append(next)

	return -1
