extends CanvasLayer
class_name HUD

# --------------------------------------------------
# HUD Controller
# Manages UI elements, status panel, buttons, and toasts.
# Decoupled from RL algorithms.
# --------------------------------------------------

signal train_requested
signal trained_playback_requested
signal random_playback_requested
signal new_map_requested

# Top Bar
@onready var map_seed_label: Label = $TopBar/HBox/MapSeed

# Status Panel
@onready var status_panel: PanelContainer = $StatusPanel
@onready var mode_title_label: Label = $StatusPanel/VBox/ModeTitle
@onready var episode_label: Label = $StatusPanel/VBox/EpisodeLabel
@onready var progress_bar: ProgressBar = $StatusPanel/VBox/ProgressBar
@onready var stage_label: Label = $StatusPanel/VBox/StageLabel

# Training metrics
@onready var metrics_box: VBoxContainer = $StatusPanel/VBox/MetricsBox
@onready var reward_label: Label = $StatusPanel/VBox/MetricsBox/RewardLabel
@onready var avg_reward_label: Label = $StatusPanel/VBox/MetricsBox/AvgRewardLabel
@onready var epsilon_label: Label = $StatusPanel/VBox/MetricsBox/EpsilonLabel
@onready var states_label: Label = $StatusPanel/VBox/MetricsBox/StatesLabel

# Playback metrics
@onready var playback_box: VBoxContainer = $StatusPanel/VBox/PlaybackBox
@onready var playback_step_label: Label = $StatusPanel/VBox/PlaybackBox/StepLabel
@onready var playback_reward_label: Label = $StatusPanel/VBox/PlaybackBox/PlaybackRewardLabel
@onready var playback_collected_label: Label = $StatusPanel/VBox/PlaybackBox/CollectedLabel
# Node still exists in the scene, but the "Remaining: N" line is no longer shown.
@onready var playback_remaining_label: Label = $StatusPanel/VBox/PlaybackBox/RemainingLabel

# Evaluation summary
@onready var eval_box: VBoxContainer = $StatusPanel/VBox/EvalBox
@onready var eval_summary_label: Label = $StatusPanel/VBox/EvalBox/EvalSummaryLabel

# Reward Graph
@onready var reward_graph: RewardGraph = $RewardGraphPanel/VBox/RewardGraph

# Controls
@onready var train_button: Button = $Controls/TrainButton
@onready var trained_button: Button = $Controls/TrainedButton
@onready var random_button: Button = $Controls/RandomButton
@onready var new_map_button: Button = $Controls/NewMapButton

# Toast
@onready var toast_panel: PanelContainer = $Toast
@onready var toast_label: Label = $Toast/ToastLabel

var toast_timer: SceneTreeTimer = null
var current_total_episodes: int = 5000
var total_collectibles_in_map: int = 0
var states_learned: int = 0


func _ready() -> void:
	train_button.pressed.connect(_on_train_button_pressed)
	trained_button.pressed.connect(_on_trained_button_pressed)
	random_button.pressed.connect(_on_random_button_pressed)
	new_map_button.pressed.connect(_on_new_map_button_pressed)

	toast_panel.visible = false
	show_idle_state()


# --------------------------------------------------
# View States
# --------------------------------------------------


func show_idle_state() -> void:
	mode_title_label.text = "READY"
	episode_label.visible = false
	progress_bar.visible = false
	stage_label.visible = false

	metrics_box.visible = true
	reward_label.visible = false
	avg_reward_label.visible = false
	epsilon_label.visible = false
	states_label.text = "States learned: %d" % states_learned
	states_label.visible = true

	playback_box.visible = false
	eval_box.visible = false
	set_controls_enabled(true)


func set_map_seed(seed_val: int) -> void:
	map_seed_label.text = "MAP %d" % seed_val


func set_states_learned(count: int) -> void:
	states_learned = count
	states_label.text = "States learned: %d" % states_learned


# --------------------------------------------------
# Training Lifecycle
# --------------------------------------------------


func on_training_started(total_episodes: int) -> void:
	current_total_episodes = total_episodes
	reward_graph.clear()

	mode_title_label.text = "TRAINING"
	episode_label.text = "Episode 0 / %s" % format_number(total_episodes)
	episode_label.visible = true

	progress_bar.min_value = 0
	progress_bar.max_value = total_episodes
	progress_bar.value = 0
	progress_bar.visible = true

	stage_label.text = "Training..."
	stage_label.visible = true

	metrics_box.visible = true
	reward_label.text = "Reward: --"
	reward_label.visible = true
	avg_reward_label.text = "Avg reward: --"
	avg_reward_label.visible = true
	epsilon_label.text = "Epsilon: 1.00"
	epsilon_label.visible = true
	states_label.text = "States learned: %d" % states_learned
	states_label.visible = true

	playback_box.visible = false
	eval_box.visible = false

	set_controls_enabled(false)
	show_toast("Training started (%s episodes)" % format_number(total_episodes), 2.0)


func on_training_progress(data: Dictionary) -> void:
	var ep: int = data.get("episode", 0)
	var total: int = data.get("total_episodes", current_total_episodes)
	var reward: float = data.get("reward", 0.0)
	var eps: float = data.get("epsilon", 0.0)
	var states: int = data.get("states_count", states_learned)
	states_learned = states

	episode_label.text = "Episode %s / %s" % [format_number(ep), format_number(total)]
	progress_bar.value = ep
	stage_label.text = "Training..."

	reward_label.text = "Reward: %+.1f" % reward
	epsilon_label.text = "Epsilon: %.2f" % eps
	states_label.text = "States learned: %d" % states_learned

	# Feed every episode's reward since the last snapshot into the graph,
	# not just the reward from this one snapshot episode. Otherwise the
	# graph only ever gets one point per visual_training_interval batch
	# (e.g. 80 points for 2000 episodes at an interval of 25) instead of
	# one point per actual training episode.
	var batch: Array = data.get("episode_rewards_batch", [reward])
	if batch.is_empty():
		batch = [reward]
	var batch_start_ep: int = data.get("batch_start_episode", ep - batch.size() + 1)
	for i in range(batch.size()):
		reward_graph.add_reward(batch_start_ep + i, batch[i])

	if reward_graph.moving_averages.size() > 0:
		var current_avg: float = reward_graph.moving_averages[-1]
		avg_reward_label.text = "Avg reward: %+.1f" % current_avg


func on_training_evaluating() -> void:
	stage_label.text = "Evaluating held-out maps..."
	progress_bar.value = progress_bar.max_value


func on_training_finished(data: Dictionary) -> void:
	var total: int = data.get("total_episodes", current_total_episodes)
	var states: int = data.get("states_learned", states_learned)
	states_learned = states

	mode_title_label.text = "TRAINING COMPLETE"
	episode_label.text = "Episodes: %s" % format_number(total)
	progress_bar.visible = false
	stage_label.text = "Complete"

	states_label.text = "States learned: %d" % states_learned

	if reward_graph.moving_averages.size() > 0:
		var final_avg: float = reward_graph.moving_averages[-1]
		avg_reward_label.text = "Final avg reward: %+.1f" % final_avg

	set_controls_enabled(true)
	show_toast("✓ Training complete | Q-table saved", 3.0)


func on_evaluation_finished(results: Array[Dictionary]) -> void:
	eval_box.visible = true
	var summary_text := ""
	var success_count: int = 0

	for res in results:
		var seed_num: int = res.get("seed", 0)
		var reward_val: float = res.get("reward", 0.0)
		var completed: bool = res.get("completed", false)
		var status_icon := "✓" if completed else "✗"
		if completed:
			success_count += 1
		summary_text += "Map %d: %+.1f %s\n" % [seed_num, reward_val, status_icon]

	summary_text += "Success: %d / %d" % [success_count, results.size()]
	eval_summary_label.text = summary_text


# --------------------------------------------------
# Playback Lifecycle
# --------------------------------------------------


func on_playback_started(policy_name: String, total_collectibles: int) -> void:
	total_collectibles_in_map = total_collectibles

	mode_title_label.text = policy_name.to_upper() + " POLICY"
	episode_label.visible = false
	progress_bar.visible = false
	stage_label.visible = false
	metrics_box.visible = false
	eval_box.visible = false

	playback_box.visible = true
	playback_step_label.text = "Step: 0"
	playback_reward_label.text = "Reward: +0.0"
	playback_collected_label.text = "Collected: 0 / %d" % total_collectibles_in_map
	playback_remaining_label.visible = false

	show_toast("▶ %s playback started" % policy_name, 2.0)


func on_playback_step(data: Dictionary) -> void:
	var step: int = data.get("step", 0)
	var total_reward: float = data.get("total_reward", 0.0)
	var remaining: int = data.get("remaining", 0)
	var collected: int = max(0, total_collectibles_in_map - remaining)

	playback_step_label.text = "Step: %d" % step
	playback_reward_label.text = "Reward: %+.1f" % total_reward
	playback_collected_label.text = "Collected: %d / %d" % [collected, total_collectibles_in_map]


func on_playback_finished(data: Dictionary) -> void:
	var policy_name: String = data.get("policy_name", "Playback")
	var completed: bool = data.get("completed", false)
	var steps: int = data.get("steps", 0)
	var reward: float = data.get("reward", 0.0)

	var status_msg := "✓ Target reached" if completed else "Stopped"
	show_toast("%s finished: %s (%d steps, %+.1f reward)" % [policy_name, status_msg, steps, reward], 3.0)


# --------------------------------------------------
# Controls and Buttons
# --------------------------------------------------


func set_controls_enabled(enabled: bool) -> void:
	train_button.disabled = not enabled
	trained_button.disabled = not enabled
	random_button.disabled = not enabled
	new_map_button.disabled = not enabled


func _on_train_button_pressed() -> void:
	train_requested.emit()


func _on_trained_button_pressed() -> void:
	trained_playback_requested.emit()


func _on_random_button_pressed() -> void:
	random_playback_requested.emit()


func _on_new_map_button_pressed() -> void:
	new_map_requested.emit()


# --------------------------------------------------
# Toast Notification System
# --------------------------------------------------


func show_toast(message: String, duration: float = 2.5) -> void:
	toast_label.text = message
	toast_panel.visible = true
	toast_panel.modulate.a = 1.0

	var tween := create_tween()
	tween.tween_interval(duration)
	tween.tween_property(toast_panel, "modulate:a", 0.0, 0.4)
	tween.tween_callback(func(): toast_panel.visible = false)


# --------------------------------------------------
# Utility
# --------------------------------------------------


func format_number(value: int) -> String:
	var str_val := str(abs(value))
	var formatted := ""
	var count := 0
	for i in range(str_val.length() - 1, -1, -1):
		if count > 0 and count % 3 == 0:
			formatted = "," + formatted
		formatted = str_val[i] + formatted
		count += 1
	return ("-" if value < 0 else "") + formatted
