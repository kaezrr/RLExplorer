extends Node
class_name GridWorld

# V3 environment specification.
#
# The public API intentionally remains compatible with the existing Godot
# project: generate_map(seed) returns a grid Array and find_agent_start()
# locates the AGENT_START marker in that grid.

const GRID_SIZE := 12

const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3

const NUM_COLLECTIBLES := 6
const OBSTACLE_DENSITY := 0.18
const MAX_GENERATION_ATTEMPTS := 1000

const DIRECTIONS := [
	Vector2i.UP,
	Vector2i.DOWN,
	Vector2i.LEFT,
	Vector2i.RIGHT,
]


func generate_map(map_seed: int) -> Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = map_seed

	for _attempt in range(MAX_GENERATION_ATTEMPTS):
		var grid := generate_raw_map(rng)
		var start_position := find_agent_start(grid)

		var reachability := Reachability.new()
		if reachability.is_map_valid(grid, start_position):
			return grid

	# This should be extremely unlikely for the configured 12x12/18%
	# environment, but return a guaranteed-valid map rather than breaking the
	# running demo if a pathological RNG stream is ever encountered.
	push_warning(
		"Could not generate a valid V3 map after %d attempts; using a fallback map."
		% MAX_GENERATION_ATTEMPTS
	)
	return generate_fallback_map(rng)


func generate_raw_map(rng: RandomNumberGenerator) -> Array:
	var grid: Array = []

	# Match the V3 environment semantics: every cell independently becomes an
	# obstacle with probability OBSTACLE_DENSITY.
	for y in range(GRID_SIZE):
		var row: Array = []
		for x in range(GRID_SIZE):
			row.append(OBSTACLE if rng.randf() < OBSTACLE_DENSITY else EMPTY)
		grid.append(row)

	# Build the walkable set after obstacle generation. The start and all
	# collectibles are sampled from distinct walkable cells.
	var walkable: Array[Vector2i] = []
	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			if grid[y][x] == EMPTY:
				walkable.append(Vector2i(x, y))

	# A 12x12 grid at this obstacle density should always have enough space,
	# but keep this guard so the method is safe if the constants are changed.
	if walkable.size() < NUM_COLLECTIBLES + 1:
		return generate_fallback_map(rng)

	var start_index := rng.randi_range(0, walkable.size() - 1)
	var start_position: Vector2i = walkable[start_index]
	walkable.remove_at(start_index)
	grid[start_position.y][start_position.x] = AGENT_START

	for _i in range(NUM_COLLECTIBLES):
		var item_index := rng.randi_range(0, walkable.size() - 1)
		var item_position: Vector2i = walkable[item_index]
		walkable.remove_at(item_index)
		grid[item_position.y][item_position.x] = COLLECTIBLE

	return grid


func generate_fallback_map(rng: RandomNumberGenerator) -> Array:
	# Guaranteed-valid emergency map. It preserves the same logical contract
	# (12x12, one start, six collectibles, no obstacles) without touching the
	# renderer or any downstream code.
	var grid: Array = []
	for y in range(GRID_SIZE):
		var row: Array = []
		for x in range(GRID_SIZE):
			row.append(EMPTY)
		grid.append(row)

	var cells: Array[Vector2i] = []
	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			cells.append(Vector2i(x, y))

	var start_index := rng.randi_range(0, cells.size() - 1)
	var start_position: Vector2i = cells[start_index]
	cells.remove_at(start_index)
	grid[start_position.y][start_position.x] = AGENT_START

	for _i in range(NUM_COLLECTIBLES):
		var item_index := rng.randi_range(0, cells.size() - 1)
		var item_position: Vector2i = cells[item_index]
		cells.remove_at(item_index)
		grid[item_position.y][item_position.x] = COLLECTIBLE

	return grid


func find_agent_start(grid: Array) -> Vector2i:
	for y in range(grid.size()):
		for x in range(grid[y].size()):
			if grid[y][x] == AGENT_START:
				return Vector2i(x, y)

	return Vector2i.ZERO


func get_collectibles(grid: Array) -> Array[Vector2i]:
	var collectibles: Array[Vector2i] = []

	for y in range(grid.size()):
		for x in range(grid[y].size()):
			if grid[y][x] == COLLECTIBLE:
				collectibles.append(Vector2i(x, y))

	return collectibles


func print_map(grid: Array) -> void:
	print("========== GRID MAP ==========")

	for y in range(GRID_SIZE):
		var line := ""

		for x in range(GRID_SIZE):
			match grid[y][x]:
				EMPTY:
					line += ". "
				OBSTACLE:
					line += "# "
				COLLECTIBLE:
					line += "C "
				AGENT_START:
					line += "A "

		print(line)

	print("==============================")
