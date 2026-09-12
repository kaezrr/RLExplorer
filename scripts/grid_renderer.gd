extends Node3D
class_name GridRenderer

const EMPTY := 0
const OBSTACLE := 1
const COLLECTIBLE := 2
const AGENT_START := 3

const FLOOR_LIGHT_ITEM := 0
const OBSTACLE_ITEM := 1
const COLLECTIBLE_ITEM := 2
const FLOOR_DARK_ITEM := 3

const TILE_SIZE := 1.0
const AGENT_Y_OFFSET := 1.0


func render_grid(grid_data: Array, grid_map: GridMap) -> void:
	grid_map.clear()

	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):
			var cell_type: int = grid_data[y][x]

			var grid_position := Vector3i(x, 0, y)

			# Always place a floor underneath every cell, alternating shade
			# by (x + y) parity so individual cells are visually distinct.
			var floor_item := FLOOR_LIGHT_ITEM if (x + y) % 2 == 0 else FLOOR_DARK_ITEM
			grid_map.set_cell_item(
				grid_position,
				floor_item
			)

			# Add the object occupying the cell.
			match cell_type:
				OBSTACLE:
					grid_map.set_cell_item(
						grid_position + Vector3i(0, 1, 0),
						OBSTACLE_ITEM
					)

				COLLECTIBLE:
					grid_map.set_cell_item(
						grid_position + Vector3i(0, 1, 0),
						COLLECTIBLE_ITEM
					)

				AGENT_START:
					# The agent is rendered separately.
					pass


func grid_to_world(grid_position: Vector2i) -> Vector3:
	return Vector3(
		grid_position.x * TILE_SIZE + TILE_SIZE * 0.5,
		AGENT_Y_OFFSET,
		grid_position.y * TILE_SIZE + TILE_SIZE * 0.5
	)
