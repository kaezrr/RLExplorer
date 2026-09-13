extends RefCounted
class_name QLearning

const ACTION_COUNT = 4

# Bumped whenever GridAgent.get_state_key() changes shape/meaning.
# A saved table is only meaningful for the state encoding that
# produced it - loading a table from an older, incompatible
# encoding would silently mix up unrelated states and actions.
const STATE_FORMAT_VERSION := 2

var q_table: Dictionary = {}
var rng := RandomNumberGenerator.new()


func _init() -> void:
	rng.randomize()


func ensure_state(state_key: String) -> void:
	if not q_table.has(state_key):
		q_table[state_key] = [0.0, 0.0, 0.0, 0.0]


func get_action(state_key: String, epsilon: float) -> int:
	ensure_state(state_key)

	if rng.randf() < epsilon:
		return rng.randi_range(0, ACTION_COUNT - 1)

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


func save_q_table(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)

	if file == null:
		print("Failed to open Q-table for writing: ", path)
		return false

	var payload := {
		"version": STATE_FORMAT_VERSION,
		"q_table": q_table
	}

	var json_text := JSON.stringify(payload)
	file.store_string(json_text)
	file.close()

	print("Q-table saved: ", path)
	print("Q-table format version: ", STATE_FORMAT_VERSION)
	print("Q-table states saved: ", q_table.size())

	return true


func load_q_table(path: String) -> bool:
	if not FileAccess.file_exists(path):
		print("Q-table file does not exist: ", path)
		return false

	var file := FileAccess.open(path, FileAccess.READ)

	if file == null:
		print("Failed to open Q-table for reading: ", path)
		return false

	var json_text := file.get_as_text()
	file.close()

	var json := JSON.new()
	var parse_error := json.parse(json_text)

	if parse_error != OK:
		print("Failed to parse Q-table JSON: ", json.get_error_message())
		return false

	if not json.data is Dictionary:
		print("Invalid Q-table format: expected Dictionary.")
		return false

	var payload: Dictionary = json.data

	# Reject tables saved before versioning existed, or saved by a
	# different state encoding. Loading these would map today's
	# state keys onto stale, meaningless Q-values instead of
	# failing loudly - so we fail loudly instead.
	if not payload.has("version") or not payload.has("q_table"):
		print("Q-table file is from an older, unversioned format.")
		print("It is not compatible with the current state encoding.")
		print("Please retrain (press T) to generate a new Q-table.")
		return false

	if payload["version"] != STATE_FORMAT_VERSION:
		print(
			"Q-table version mismatch: file is version ",
			payload["version"],
			", current encoding is version ",
			STATE_FORMAT_VERSION,
			"."
		)
		print("Please retrain (press T) to generate a new Q-table.")
		return false

	var loaded_table = payload["q_table"]

	if not loaded_table is Dictionary:
		print("Invalid Q-table format: expected Dictionary.")
		return false

	for state_key in loaded_table:
		var q_values = loaded_table[state_key]

		if not q_values is Array:
			print("Invalid Q-table state: ", state_key)
			return false

		if q_values.size() != ACTION_COUNT:
			print("Invalid Q-values for state: ", state_key)
			return false

	q_table = loaded_table

	print("Q-table loaded: ", path)
	print("Q-table format version: ", payload["version"])
	print("Q-table states loaded: ", q_table.size())

	return true
