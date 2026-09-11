extends Node
class_name GridWorld

## Block 1: Logical grid-world representation and procedural generation.
##
## Cell types:
## 0 = Empty
## 1 = Obstacle
## 2 = Collectible
## 3 = Agent Start
##
## The generator is deterministic:
## generate_map(42) will always produce the same map.

const GRID_SIZE: int = 12

const EMPTY: int = 0
const OBSTACLE: int = 1
const COLLECTIBLE: int = 2
const AGENT_START: int = 3

const MIN_COLLECTIBLES: int = 5
const MAX_COLLECTIBLES: int = 8

# Approximately 15–20% of the grid.
const OBSTACLE_DENSITY: float = 0.18

# Generated map.
var grid: Array = []

# Starting position of the agent.
var agent_start: Vector2i = Vector2i.ZERO

# Seed used to generate the current map.
var current_seed: int = 0


## Generates a deterministic 12x12 map from the supplied seed.
##
## Calling:
##     generate_map(42)
## twice will produce exactly the same grid.
func generate_map(seed_value: int) -> Array:
	current_seed = seed_value

	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value

	# Start with an empty grid.
	grid.clear()

	for y in range(GRID_SIZE):
		var row: Array = []

		for x in range(GRID_SIZE):
			row.append(EMPTY)

		grid.append(row)

	# ---------------------------------------------------------
	# 1. Place obstacles.
	# ---------------------------------------------------------

	var total_cells := GRID_SIZE * GRID_SIZE
	var obstacle_count := int(total_cells * OBSTACLE_DENSITY)

	var available_cells: Array[Vector2i] = []

	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			available_cells.append(Vector2i(x, y))

	# Shuffle deterministically using our seeded RNG.
	available_cells.shuffle()

	for i in range(obstacle_count):
		var position: Vector2i = available_cells[i]
		grid[position.y][position.x] = OBSTACLE

	# ---------------------------------------------------------
	# 2. Pick an empty cell for the agent start.
	# ---------------------------------------------------------

	var empty_cells: Array[Vector2i] = get_empty_cells()

	if empty_cells.is_empty():
		push_error("Could not find an empty cell for the agent start.")
		return grid

	var start_index := rng.randi_range(0, empty_cells.size() - 1)
	agent_start = empty_cells[start_index]

	grid[agent_start.y][agent_start.x] = AGENT_START

	# ---------------------------------------------------------
	# 3. Place collectibles.
	# ---------------------------------------------------------

	empty_cells = get_empty_cells()

	if empty_cells.is_empty():
		push_error("Could not find cells for collectibles.")
		return grid

	var collectible_count := rng.randi_range(
		MIN_COLLECTIBLES,
		MAX_COLLECTIBLES
	)

	# Don't request more collectibles than available cells.
	collectible_count = min(collectible_count, empty_cells.size())

	for i in range(collectible_count):
		var collectible_index := rng.randi_range(
			0,
			empty_cells.size() - 1
		)

		var collectible_position: Vector2i = empty_cells[collectible_index]

		grid[collectible_position.y][collectible_position.x] = COLLECTIBLE

		# Remove this cell so another collectible cannot be placed there.
		empty_cells.remove_at(collectible_index)

	# ---------------------------------------------------------
	# 4. Print the generated map for debugging.
	# ---------------------------------------------------------

	print_map()

	return grid


## Returns all currently empty cells.
func get_empty_cells() -> Array[Vector2i]:
	var cells: Array[Vector2i] = []

	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			if grid[y][x] == EMPTY:
				cells.append(Vector2i(x, y))

	return cells


## Returns true if the supplied grid position is inside the grid.
func is_inside_grid(position: Vector2i) -> bool:
	return (
		position.x >= 0
		and position.x < GRID_SIZE
		and position.y >= 0
		and position.y < GRID_SIZE
	)


## Returns the cell value at a position.
##
## Returns OBSTACLE for positions outside the grid.
## This makes later movement/state logic safer.
func get_cell(position: Vector2i) -> int:
	if not is_inside_grid(position):
		return OBSTACLE

	return grid[position.y][position.x]


## Changes a cell's value.
func set_cell(position: Vector2i, cell_type: int) -> void:
	if not is_inside_grid(position):
		push_warning("Attempted to set cell outside grid: ", position)
		return

	grid[position.y][position.x] = cell_type


## Returns true if a position can be occupied by the agent.
func is_walkable(position: Vector2i) -> bool:
	if not is_inside_grid(position):
		return false

	return grid[position.y][position.x] != OBSTACLE


## Returns all collectible positions.
func get_collectible_positions() -> Array[Vector2i]:
	var collectibles: Array[Vector2i] = []

	for y in range(GRID_SIZE):
		for x in range(GRID_SIZE):
			if grid[y][x] == COLLECTIBLE:
				collectibles.append(Vector2i(x, y))

	return collectibles


## Returns the number of collectibles currently remaining.
func get_collectible_count() -> int:
	return get_collectible_positions().size()


## Removes a collectible from a cell.
func collect_item(position: Vector2i) -> bool:
	if not is_inside_grid(position):
		return false

	if grid[position.y][position.x] != COLLECTIBLE:
		return false

	grid[position.y][position.x] = EMPTY
	return true


## Creates a readable text representation of the grid.
##
## Symbols:
## . = empty
## # = obstacle
## C = collectible
## A = agent start
func get_map_string() -> String:
	var output := ""

	for y in range(GRID_SIZE):
		var row_string := ""

		for x in range(GRID_SIZE):
			match grid[y][x]:
				EMPTY:
					row_string += "."
				OBSTACLE:
					row_string += "#"
				COLLECTIBLE:
					row_string += "C"
				AGENT_START:
					row_string += "A"
				_:
					row_string += "?"

		output += row_string

		if y < GRID_SIZE - 1:
			output += "\n"

	return output


## Prints the current map to the Godot Output console.
func print_map() -> void:
	print("================================")
	print("Generated Grid World")
	print("Seed: ", current_seed)
	print("Size: ", GRID_SIZE, "x", GRID_SIZE)
	print("Agent start: ", agent_start)
	print("Collectibles: ", get_collectible_count())
	print("--------------------------------")
	print(get_map_string())
	print("================================")


## Optional convenience function for testing.
##
## This allows the script to generate a map automatically
## when the node enters the scene.
func _ready() -> void:
	generate_map(42)
