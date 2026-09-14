extends Camera3D

const GRID_SIZE = GridWorld.GRID_SIZE

func _ready() -> void:
	var center := Vector3(GRID_SIZE / 2.0, 0, GRID_SIZE / 2.0)

	var horizontal_distance := GRID_SIZE * 0.9
	var height := GRID_SIZE * 1.6

	self.projection = Camera3D.PROJECTION_ORTHOGONAL
	self.size = GRID_SIZE * 1.3

	self.position = center + Vector3(horizontal_distance, height, horizontal_distance)
	self.look_at(center, Vector3.UP)

	# Push the visible content to the left, leaving room for UI on the right.
	# Value is in the same units as `size` (world units for ortho).
	self.h_offset = GRID_SIZE * 0.2
