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

@export var MAP_SEED := 42


func generate_map(map_seed: int) -> Array:
	var rng := RandomNumberGenerator.new()
	rng.seed = map_seed

	var grid: Array = []

	# Create an empty GRID_SIZE x GRID_SIZE grid.
	for y in range(GRID_SIZE):
		var row: Array = []

		for x in range(GRID_SIZE):
			row.append(EMPTY)

		grid.append(row)

	# Create a list of every grid position.
	var available_cells: Array[Vector2i] = []

	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			available_cells.append(Vector2i(x, y))

	# Randomly place obstacles using the seeded RNG.
	var total_cells := GRID_SIZE * GRID_SIZE
	var obstacle_count := int(total_cells * OBSTACLE_DENSITY)

	for i in range(obstacle_count):
		var random_index := rng.randi_range(0, available_cells.size() - 1)
		var position: Vector2i = available_cells[random_index]

		grid[position.y][position.x] = OBSTACLE

		# Remove the selected cell so it cannot be selected again.
		available_cells.remove_at(random_index)

	# The remaining cells are empty.
	# Randomly choose the number of collectibles.
	var collectible_count := rng.randi_range(
		MIN_COLLECTIBLES,
		MAX_COLLECTIBLES
	)

	for i in range(collectible_count):
		var random_index := rng.randi_range(0, available_cells.size() - 1)
		var position: Vector2i = available_cells[random_index]

		grid[position.y][position.x] = COLLECTIBLE

		# Remove it so it cannot also become the agent start.
		available_cells.remove_at(random_index)

	# Pick one remaining empty cell as the agent start.
	var start_index := rng.randi_range(0, available_cells.size() - 1)
	var start_position: Vector2i = available_cells[start_index]

	grid[start_position.y][start_position.x] = AGENT_START

	return grid


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
	

func _ready() -> void:
	var grid_a := generate_map(MAP_SEED)
	var grid_b := generate_map(MAP_SEED)

	print_map(grid_a)

	print("Same seed produces identical map: ", grid_a == grid_b)
