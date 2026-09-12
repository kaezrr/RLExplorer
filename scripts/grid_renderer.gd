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

# Assumes the agent uses Godot's DEFAULT CapsuleMesh (radius 0.5, height 2.0),
# unscaled — i.e. agent keeps a default Transform, no resizing.
# Capsule center must sit at half its height above the floor (y=0) for its
# bottom to touch the ground: 2.0 / 2 = 1.0.
const AGENT_Y_OFFSET := 1.0


func render_grid(grid_data: Array, grid_map: GridMap) -> void:
	grid_map.clear()

	for y in range(grid_data.size()):
		for x in range(grid_data[y].size()):
			var cell_type: int = grid_data[y][x]
			var grid_position := Vector3i(x, 0, y)

			# Floor cell (grid y-index 0). Checkerboard by (x + y) parity.
			# NOTE: the floor mesh's own Y transform in mesh_library.tres must be
			# offset to -thickness/2 so its TOP face lands exactly at world y=0,
			# rather than straddling y=0 (which is the default center-on-cell behavior).
			var floor_item := FLOOR_LIGHT_ITEM if (x + y) % 2 == 0 else FLOOR_DARK_ITEM
			grid_map.set_cell_item(grid_position, floor_item)

			# Objects occupy the cell one grid-index above the floor (grid y-index 1,
			# world center y=1 by this project's GridMap convention).
			match cell_type:
				OBSTACLE:
					# NOTE: the obstacle mesh's Y transform in mesh_library.tres must be
					# offset to -0.5 so a 1x1x1 cube's BOTTOM face lands at world y=0
					# (flush with the floor) instead of floating at y=0.5.
					grid_map.set_cell_item(
						grid_position + Vector3i(0, 1, 0),
						OBSTACLE_ITEM
					)

				COLLECTIBLE:
					# Left at its default mesh transform (no offset) — this is what
					# makes it float above the floor, which is the intended look.
					grid_map.set_cell_item(
						grid_position + Vector3i(0, 1, 0),
						COLLECTIBLE_ITEM
					)

				AGENT_START:
					# The agent is rendered separately via grid_to_world() below.
					pass


func grid_to_world(grid_position: Vector2i) -> Vector3:
	return Vector3(
		grid_position.x * TILE_SIZE + TILE_SIZE * 0.5,
		AGENT_Y_OFFSET,
		grid_position.y * TILE_SIZE + TILE_SIZE * 0.5
	)
