extends Control
class_name RewardGraph

# --------------------------------------------------
# Reward Graph
# Custom lightweight Control for plotting RL episode rewards.
# --------------------------------------------------

var raw_rewards: Array[float] = []
var moving_averages: Array[float] = []

@export var moving_avg_window: int = 50
@export var max_render_points: int = 400

# Color palette
@export var bg_color := Color(0.08, 0.09, 0.12, 0.7)
@export var grid_color := Color(1.0, 1.0, 1.0, 0.08)
@export var zero_line_color := Color(1.0, 1.0, 1.0, 0.25)
@export var raw_reward_color := Color(0.35, 0.65, 0.95, 0.35)
@export var moving_avg_color := Color(0.25, 0.88, 0.55, 0.95)
@export var text_color := Color(0.75, 0.78, 0.85, 0.75)


func _ready() -> void:
	resized.connect(queue_redraw)


func clear() -> void:
	raw_rewards.clear()
	moving_averages.clear()
	queue_redraw()


func add_reward(_episode: int, reward: float) -> void:
	raw_rewards.append(reward)

	var start_idx: int = max(0, raw_rewards.size() - moving_avg_window)
	var sum: float = 0.0
	var count: int = 0
	for i in range(start_idx, raw_rewards.size()):
		sum += raw_rewards[i]
		count += 1
	var avg: float = sum / float(max(1, count))
	moving_averages.append(avg)

	queue_redraw()


func set_rewards(rewards: Array) -> void:
	raw_rewards.clear()
	moving_averages.clear()

	var running_sum: float = 0.0
	for i in range(rewards.size()):
		var val: float = float(rewards[i])
		raw_rewards.append(val)
		running_sum += val
		if i >= moving_avg_window:
			running_sum -= raw_rewards[i - moving_avg_window]
			moving_averages.append(running_sum / float(moving_avg_window))
		else:
			moving_averages.append(running_sum / float(i + 1))

	queue_redraw()


func _draw() -> void:
	var rect: Rect2 = get_rect()
	var w: float = rect.size.x
	var h: float = rect.size.y

	if w <= 10.0 or h <= 10.0:
		return

	var margin_left: float = 42.0
	var margin_right: float = 12.0
	var margin_top: float = 12.0
	var margin_bottom: float = 22.0

	var plot_w: float = maxf(1.0, w - margin_left - margin_right)
	var plot_h: float = maxf(1.0, h - margin_top - margin_bottom)

	draw_rect(Rect2(margin_left, margin_top, plot_w, plot_h), bg_color)

	var font: Font = ThemeDB.fallback_font
	var font_size: int = 11

	if raw_rewards.is_empty():
		var empty_msg: String = "No reward data yet"
		var str_size: Vector2 = font.get_string_size(empty_msg, HORIZONTAL_ALIGNMENT_CENTER, -1, font_size)
		draw_string(
			font,
			Vector2(margin_left + (plot_w - str_size.x) * 0.5, margin_top + plot_h * 0.5 + 4.0),
			empty_msg,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			font_size,
			Color(text_color.r, text_color.g, text_color.b, 0.4)
		)
		return

	var min_val: float = raw_rewards[0]
	var max_val: float = raw_rewards[0]
	for val: float in raw_rewards:
		if val < min_val:
			min_val = val
		if val > max_val:
			max_val = val

	for avg: float in moving_averages:
		if avg < min_val:
			min_val = avg
		if avg > max_val:
			max_val = avg

	var padding: float = maxf(1.0, (max_val - min_val) * 0.1)
	min_val -= padding
	max_val += padding
	var val_range: float = maxf(0.001, max_val - min_val)

	for i in range(5):
		var ratio: float = float(i) / 4.0
		var y: float = margin_top + plot_h * (1.0 - ratio)
		var val_at_y: float = min_val + val_range * ratio
		draw_line(Vector2(margin_left, y), Vector2(margin_left + plot_w, y), grid_color, 1.0)

		var label: String = "%+.0f" % val_at_y
		draw_string(
			font,
			Vector2(4.0, y + 4.0),
			label,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			font_size - 1,
			text_color
		)

	if min_val <= 0.0 and max_val >= 0.0:
		var zero_ratio: float = (0.0 - min_val) / val_range
		var zero_y: float = margin_top + plot_h * (1.0 - zero_ratio)
		draw_line(
			Vector2(margin_left, zero_y),
			Vector2(margin_left + plot_w, zero_y),
			zero_line_color,
			1.5
		)

	var total_count: int = raw_rewards.size()
	var step_size: int = max(1, total_count / max_render_points)

	var to_coord: Callable = func(idx: int, val: float) -> Vector2:
		var nx: float = float(idx) / float(max(1, total_count - 1))
		var ny: float = (val - min_val) / val_range
		return Vector2(margin_left + nx * plot_w, margin_top + plot_h * (1.0 - ny))

	if total_count >= 2:
		var raw_points: PackedVector2Array = PackedVector2Array()
		for i in range(0, total_count, step_size):
			raw_points.append(to_coord.call(i, raw_rewards[i]))
		if (total_count - 1) % step_size != 0:
			raw_points.append(to_coord.call(total_count - 1, raw_rewards[total_count - 1]))

		if raw_points.size() >= 2:
			draw_polyline(raw_points, raw_reward_color, 1.0, true)

	if moving_averages.size() >= 2:
		var avg_points: PackedVector2Array = PackedVector2Array()
		for i in range(0, moving_averages.size(), step_size):
			avg_points.append(to_coord.call(i, moving_averages[i]))
		if (moving_averages.size() - 1) % step_size != 0:
			avg_points.append(
				to_coord.call(moving_averages.size() - 1, moving_averages[moving_averages.size() - 1])
			)

		if avg_points.size() >= 2:
			draw_polyline(avg_points, moving_avg_color, 2.2, true)

	var ep_text: String = "Episodes: %d →" % total_count
	draw_string(
		font,
		Vector2(margin_left + plot_w - 90.0, h - 6.0),
		ep_text,
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		font_size,
		text_color
	)
