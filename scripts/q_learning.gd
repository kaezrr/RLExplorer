extends RefCounted
class_name QLearning

## V3 linear-function-approximation Q-learning.
##
## The model does NOT maintain a state -> Q-value table.
## Instead, every action is described by the same 10-dimensional
## feature vector and one shared weight vector is used everywhere:
##
##     Q(s, a) = dot(phi(s, a), w)
##
## This mirrors test_rl_3.py's LinearQAgent.

const ACTION_COUNT := 4
const FEATURE_COUNT := 10
const WEIGHT_FORMAT_VERSION := 3

# Default V3 hyperparameters. Trainer may pass explicit values to update().
const DEFAULT_ALPHA := 0.015
const DEFAULT_GAMMA := 0.98

var weights: Array[float] = []
var rng := RandomNumberGenerator.new()
var epsilon: float = 1.0


func _init(seed: int = 0) -> void:
	weights.resize(FEATURE_COUNT)
	for i in range(FEATURE_COUNT):
		weights[i] = 0.0

	if seed != 0:
		rng.seed = seed
	else:
		rng.randomize()


func reset_weights() -> void:
	for i in range(FEATURE_COUNT):
		weights[i] = 0.0


func q_values(phi: Array) -> Array[float]:
	## phi must contain 4 rows, each containing 10 features.
	var values: Array[float] = []
	values.resize(ACTION_COUNT)

	for action in range(ACTION_COUNT):
		values[action] = _dot(phi[action])

	return values


func _dot(features: Array) -> float:
	var total := 0.0
	var count := mini(features.size(), FEATURE_COUNT)

	for i in range(count):
		total += float(features[i]) * weights[i]

	return total


func get_action(phi: Array, epsilon_value: float) -> int:
	## Epsilon-greedy action selection.
	## Ties are resolved randomly, matching numpy's tie handling in V3.
	if rng.randf() < epsilon_value:
		return rng.randi_range(0, ACTION_COUNT - 1)

	var values := q_values(phi)
	var best_value := values[0]
	var best_actions: Array[int] = [0]

	for action in range(1, ACTION_COUNT):
		var value := values[action]

		# is_equal_approx is used for the same practical purpose as
		# np.isclose() in the Python reference implementation.
		if is_equal_approx(value, best_value):
			best_actions.append(action)
		elif value > best_value:
			best_value = value
			best_actions.clear()
			best_actions.append(action)

	return best_actions[rng.randi_range(0, best_actions.size() - 1)]


func update(
	phi: Array,
	action: int,
	reward: float,
	next_phi: Array,
	done: bool,
	alpha: float = DEFAULT_ALPHA,
	gamma: float = DEFAULT_GAMMA,
) -> float:
	## One-step Q-learning update for the selected action.
	## Returns the TD error, useful for debugging/telemetry.
	if action < 0 or action >= ACTION_COUNT:
		return 0.0

	var current_q := _dot(phi[action])
	var next_q := 0.0

	if not done:
		var next_values := q_values(next_phi)
		next_q = next_values[0]
		for i in range(1, next_values.size()):
			if next_values[i] > next_q:
				next_q = next_values[i]

	var target := reward + gamma * next_q
	var td_error := target - current_q

	var selected_features: Array = phi[action]
	for i in range(FEATURE_COUNT):
		weights[i] += alpha * td_error * float(selected_features[i])

	return td_error


func set_weights(values: Array) -> bool:
	if values.size() != FEATURE_COUNT:
		print("Invalid V3 weight count: ", values.size(), " (expected ", FEATURE_COUNT, ")")
		return false

	for i in range(FEATURE_COUNT):
		weights[i] = float(values[i])

	return true


func get_weights() -> Array[float]:
	return weights.duplicate()


func save_weights(path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		print("Failed to open V3 weights for writing: ", path)
		return false

	var payload := {
		"version": WEIGHT_FORMAT_VERSION,
		"feature_count": FEATURE_COUNT,
		"weights": weights,
	}

	file.store_string(JSON.stringify(payload))
	file.close()

	print("V3 weights saved: ", path)
	print("Weight format version: ", WEIGHT_FORMAT_VERSION)
	print("Feature count: ", FEATURE_COUNT)
	print("Weights: ", weights)

	return true


func load_weights(path: String) -> bool:
	if not FileAccess.file_exists(path):
		print("V3 weight file does not exist: ", path)
		return false

	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		print("Failed to open V3 weights for reading: ", path)
		return false

	var json_text := file.get_as_text()
	file.close()

	var json := JSON.new()
	var parse_error := json.parse(json_text)
	if parse_error != OK:
		print("Failed to parse V3 weights: ", json.get_error_message())
		return false

	if not json.data is Dictionary:
		print("Invalid V3 weight format: expected Dictionary.")
		return false

	var payload: Dictionary = json.data

	if not payload.has("version") or not payload.has("feature_count") or not payload.has("weights"):
		print("V3 weight file is missing required fields.")
		return false

	if int(payload["version"]) != WEIGHT_FORMAT_VERSION:
		print(
			"V3 weight version mismatch: file is version ",
			payload["version"],
			", current format is version ",
			WEIGHT_FORMAT_VERSION,
			".",
		)
		return false

	if int(payload["feature_count"]) != FEATURE_COUNT:
		print(
			"V3 feature-count mismatch: file has ",
			payload["feature_count"],
			", current agent expects ",
			FEATURE_COUNT,
			".",
		)
		return false

	var loaded_weights = payload["weights"]
	if not loaded_weights is Array:
		print("Invalid V3 weight format: weights must be an Array.")
		return false

	if loaded_weights.size() != FEATURE_COUNT:
		print("Invalid V3 weight count: ", loaded_weights.size())
		return false

	for i in range(FEATURE_COUNT):
		if not (loaded_weights[i] is float or loaded_weights[i] is int):
			print("Invalid V3 weight at index ", i, ": ", loaded_weights[i])
			return false

	return set_weights(loaded_weights)
