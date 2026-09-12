extends Node
class_name Reachability


const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3


const DIRECTIONS := [
	Vector2i.UP,
	Vector2i.DOWN,
	Vector2i.LEFT,
	Vector2i.RIGHT
]


func get_reachable_cells(
	grid_data: Array,
	start_pos: Vector2i
) -> Array[Vector2i]:

	var reachable: Array[Vector2i] = []
	var visited := {}

	var queue: Array[Vector2i] = []

	queue.append(start_pos)
	visited[start_pos] = true

	while not queue.is_empty():
		var current: Vector2i = queue.pop_front()

		reachable.append(current)

		for direction in DIRECTIONS:
			var next: Vector2i = current + direction

			# Ignore positions outside the grid.
			if not is_inside_grid(grid_data, next):
				continue

			# Ignore obstacles.
			if grid_data[next.y][next.x] == OBSTACLE:
				continue

			# Ignore cells we already visited.
			if visited.has(next):
				continue

			visited[next] = true
			queue.append(next)

	return reachable


func is_map_valid(
	grid_data: Array,
	start_pos: Vector2i
) -> bool:

	var reachable := get_reachable_cells(
		grid_data,
		start_pos
	)

	var reachable_set := {}

	for position in reachable:
		reachable_set[position] = true

	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):

			if grid_data[y][x] == COLLECTIBLE:
				var collectible_position := Vector2i(x, y)

				if not reachable_set.has(collectible_position):
					return false

	return true


func is_inside_grid(
	grid_data: Array,
	position: Vector2i
) -> bool:

	if position.y < 0:
		return false

	if position.y >= grid_data.size():
		return false

	if position.x < 0:
		return false

	if position.x >= grid_data[position.y].size():
		return false

	return true
