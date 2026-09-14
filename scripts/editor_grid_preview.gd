@tool
extends GridMap

## Editor-only placeholder grid, purely for visual debugging.
## Has zero effect when the actual game runs — everything here
## is guarded behind Engine.is_editor_hint().

@export var debug_seed := 1
@export var show_placeholder := true:
	set(value):
		show_placeholder = value
		if Engine.is_editor_hint():
			_refresh_placeholder()

func _ready() -> void:
	if not Engine.is_editor_hint():
		return
	_refresh_placeholder()

func _refresh_placeholder() -> void:
	if not Engine.is_editor_hint():
		return

	clear()

	if not show_placeholder:
		return

	var world := GridWorld.new()
	var renderer := GridRenderer.new()

	add_child(world)
	add_child(renderer)

	var grid := world.generate_map(debug_seed)
	renderer.render_grid(grid, self)

	world.queue_free()
	renderer.queue_free()
