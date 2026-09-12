extends Node
class_name GridWorld


const GRID_SIZE := 12

const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3

const MIN_COLLECTIBLES := 5
const MAX_COLLECTIBLES := 8

const OBSTACLE_DENSITY := 0.18
const MAX_GENERATION_ATTEMPTS := 50


func generate_map(map_seed: int) -> Array:
	var current_seed := map_seed

	for attempt in range(MAX_GENERATION_ATTEMPTS):
		var grid := generate_raw_map(current_seed)

		var start_position := find_agent_start(grid)

		var reachability := Reachability.new()

		if reachability.is_map_valid(
			grid,
			start_position
		):
			print(
				"Valid map generated with seed ",
				current_seed,
				" after ",
				attempt + 1,
				" attempt(s)."
			)

			return grid

		current_seed += 1

	print(
		"WARNING: Could not generate a valid map after ",
		MAX_GENERATION_ATTEMPTS,
		" attempts."
	)

	# Fallback.
	return generate_raw_map(current_seed)


func generate_raw_map(map_seed: int) -> Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = map_seed

	var grid: Array = []

	# --------------------------------------------------
	# 1. Create an empty grid.
	# --------------------------------------------------

	for y in range(GRID_SIZE):
		var row: Array = []

		for x in range(GRID_SIZE):
			row.append(EMPTY)

		grid.append(row)


	# --------------------------------------------------
	# 2. Create a list of every cell.
	# --------------------------------------------------

	var available_cells: Array[Vector2i] = []

	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			available_cells.append(
				Vector2i(x, y)
			)


	# --------------------------------------------------
	# 3. Place obstacles.
	# --------------------------------------------------

	var total_cells := GRID_SIZE * GRID_SIZE

	var obstacle_count := int(
		total_cells * OBSTACLE_DENSITY
	)

	for i in range(obstacle_count):
		var random_index := rng.randi_range(
			0,
			available_cells.size() - 1
		)

		var position: Vector2i = available_cells[
			random_index
		]

		grid[position.y][position.x] = OBSTACLE

		available_cells.remove_at(
			random_index
		)


	# --------------------------------------------------
	# 4. Place collectibles.
	# --------------------------------------------------

	var collectible_count := rng.randi_range(
		MIN_COLLECTIBLES,
		MAX_COLLECTIBLES
	)

	for i in range(collectible_count):
		var random_index := rng.randi_range(
			0,
			available_cells.size() - 1
		)

		var position: Vector2i = available_cells[
			random_index
		]

		grid[position.y][position.x] = COLLECTIBLE

		available_cells.remove_at(
			random_index
		)


	# --------------------------------------------------
	# 5. Place the agent start.
	# --------------------------------------------------

	var start_index := rng.randi_range(
		0,
		available_cells.size() - 1
	)

	var start_position: Vector2i = available_cells[
		start_index
	]

	grid[start_position.y][start_position.x] = AGENT_START


	return grid


func find_agent_start(grid: Array) -> Vector2i:
	for y in range(grid.size()):
		for x in range(grid[y].size()):

			if grid[y][x] == AGENT_START:
				return Vector2i(x, y)

	return Vector2i.ZERO


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
