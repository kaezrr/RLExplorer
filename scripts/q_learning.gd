extends RefCounted
class_name QLearning


const ACTION_COUNT = 4

var q_table: Dictionary = {}

var rng := RandomNumberGenerator.new()


func _init() -> void:
	rng.randomize()


func ensure_state(state_key: String) -> void:
	if not q_table.has(state_key):
		q_table[state_key] = [0.0, 0.0, 0.0, 0.0]


func get_action(state_key: String, epsilon: float) -> int:
	ensure_state(state_key)

	# Exploration
	if rng.randf() < epsilon:
		return rng.randi_range(0, ACTION_COUNT - 1)

	# Exploitation
	var q_values: Array = q_table[state_key]

	var best_action := 0
	var best_q_value: float = q_values[0]

	for action in range(1, ACTION_COUNT):
		if q_values[action] > best_q_value:
			best_q_value = q_values[action]
			best_action = action

	return best_action


func update(
	state_key: String,
	action: int,
	reward: float,
	next_state_key: String,
	alpha: float,
	gamma: float
) -> void:
	ensure_state(state_key)
	ensure_state(next_state_key)

	var q_values: Array = q_table[state_key]
	var next_q_values: Array = q_table[next_state_key]

	var best_next_q: float = next_q_values[0]

	for next_q in next_q_values:
		if next_q > best_next_q:
			best_next_q = next_q

	var td_target: float = reward + gamma * best_next_q

	var td_error: float = td_target - q_values[action]

	q_values[action] += alpha * td_error

	q_table[state_key] = q_values
