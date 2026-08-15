# Mid-Gesture Rock-Paper-Scissors AI

An interactive first-to-three student-fair game that predicts a visitor's
rock-paper-scissors move while their fingers are still moving. MediaPipe tracks one hand,
a small PyTorch MLP classifies each partial pose, and the game commits to a counter-move
before the final reveal. As soon as it locks, it declares the gesture it expects so the
visitor can hold course or try to fool it by switching. The default display is game-first;
a live graph of the MLP's activations and weighted connections is available as an alternate
view.

The application stores no booth camera frames. The training capture command writes only
landmark trajectories and pseudonymous session metadata.

## Requirements

- Apple Silicon Mac for the verified booth target
- A webcam and macOS camera permission for the terminal application
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.12, which `uv` installs automatically when necessary

Do not install dependencies with `pip`. The project environment and lockfile are managed
by `uv`.

## Initial setup

```bash
uv sync --locked
uv run --locked rps-setup
uv run --locked rps-preflight --allow-untrained
```

`rps-setup` downloads the pinned official MediaPipe Hand Landmarker task and verifies its
SHA-256 checksum. The asset is kept under `assets/` and is ignored by Git.

If preflight reports `not authorized to capture video`, open **System Settings > Privacy &
Security > Camera**, enable the terminal or Codex application running the command, then rerun
preflight.

The preflight command verifies Python, MediaPipe initialization, the webcam, checkpoint,
and warmed-up batch-one latency on CPU and Apple MPS. MPS is selected only if its median
end-to-end inference latency is at least 10% below CPU. Use `--skip-camera` when running
preflight without a webcam.

## Run the demo

```bash
uv run --locked rps-demo --camera 0 --device auto
```

Add `--fullscreen` for the booth display or `--mute` to start without audio. The game uses
short generated WAV cues through macOS `afplay`; missing audio support degrades silently to
visual-only play. Controls are:

- `N`: toggle the game and neural-network views
- `M`: mute or unmute audio
- `R`: reset current match points
- `C`: clear all session scores and match totals
- `Q` or Escape: exit

Without `models/gesture_mlp.pt`, the application intentionally opens in a prominent
`UNTRAINED - VISUALIZATION MODE`: random activations are shown, but gameplay and scoring
are disabled. This makes UI development possible without presenting random predictions as
a trained model.

Each trained round follows this timeline:

```text
250 ms centered closed-fist hold
ROCK -> PAPER -> SCISSORS at 600 ms intervals, then SHOOT!
150 ms after SHOOT: continuous partial-pose predictions begin
200-450 ms: prediction is visibly declared and the AI response locks
650-950 ms: held final pose is scored
950 ms: locked AI response and result are revealed
1.2 s result celebration, then a 200 ms hand-clear before the next round
```

The screen freezes the prediction and its lock time immediately. The AI response is already
committed but stays face-down until scoring. Changing gestures after the declaration is
allowed: a successful switch is celebrated as fooling the AI. Wins award one point and ties
award no point. The first player to three wins the match; the match result remains for 2.5
seconds and then resets after the hand leaves. Session round, streak, tie, and match totals
remain in memory until `C` is pressed or the app exits.

## Collect training trajectories

Collect data from 8-12 people; ten participants with 20 repetitions per class produces the
target 600 trajectories. Use a non-identifying participant alias:

```bash
uv run --locked rps-capture --participant P01 --repetitions 20 --camera 0
```

The capture UI randomizes prompts, prevents three identical prompts in a row, and repeats
trials without at least four tracked frames in both the early and final windows. The mirrored
preview draws all 21 tracked joints and their connections. If the skeleton does not follow
the participant's hand, repeat the trial rather than saving a bad trajectory.

Files are written atomically beneath `data/landmarks/<participant>/<session>/` and contain:

- 21 x 3 landmarks and relative timestamps
- Handedness
- Prompted class
- Participant alias, session ID, trajectory ID, and capture configuration

No image or video encoding exists in the capture path. The complete data directory is
ignored by Git.

## Review incorrect labels

The prompted gesture is stored in each capture, but a participant may accidentally perform
another pose. Review the landmark-only animation before training:

```bash
uv run --locked rps-review-data
```

The reviewer never displays or writes camera frames. It animates the 21 captured landmarks
and saves an atomic `data/landmarks/review-manifest.json` overlay after every decision:

- `R`, `P`, or `S`: assign the observed final pose; selecting the prompt records `keep`
- `K`: keep the prompted label
- `X`: exclude an unclear, incomplete, or mixed trajectory
- `U`: clear the current review
- Left/right arrows: navigate; Space: pause; `Q` or Escape: quit

By default, already reviewed trajectories are skipped and the promoted
`models/gesture_mlp.pt` provides model assistance. The UI shows the animated frame prediction,
the averaged final-window probabilities, prediction stability, and prompt/model disagreements.
Use `--no-model` for an independent human-only pass.

If a high-confidence model prediction conflicts with a chosen human label, the review is
saved but auto-advance pauses. Inspect the skeleton, then press Right to accept the human
decision or press another label key. The model never changes a label automatically.

Use `--all` to revisit reviewed trajectories, or filter with `--participant P01` and
`--session <session-id>`. Model assistance can also prioritize likely mistakes:

```bash
uv run --locked rps-review-data --suspicious-first
uv run --locked rps-review-data --suspicious-only --participant P01
```

Model disagreement only changes review order; every relabel or exclusion requires a human
keypress. Training and evaluation apply the manifest automatically, report review counts,
and include it in the dataset fingerprint. Raw `.npz` captures remain unchanged. If only
deployment frames resemble another gesture but the held final pose matches the prompt, keep
the trajectory: those partial poses are intentional mid-gesture examples.

## Train and evaluate

```bash
uv run --locked rps-train --device auto
uv run --locked rps-evaluate --split test --device auto
```

Training uses participant-disjoint train, validation, and test splits. Six frames are
selected per trajectory near 250, 350, 450, 700, 800, and 900 ms, giving early and final
poses equal weight. Training applies small landmark rotations and coordinate jitter.

The fixed network is:

```text
63 -> Linear(16) -> ReLU -> Linear(8) -> ReLU -> Linear(3)
```

The validated default recipe uses participant/class/phase-balanced sampling, AdamW, cosine
learning-rate decay, and validation early stopping over at most 160 selection epochs. It then
refits from scratch on train plus validation participants for the selected epoch count while
leaving the test participant untouched. A scalar temperature fitted on the selection
validation split and 99th-percentile activation scales are stored in the checkpoint.

Training writes a JSON report under `reports/` and promotes the model to
`models/gesture_mlp.pt` only when the refitted model satisfies all booth thresholds on the
held-out test participant. Otherwise it writes `models/gesture_mlp-candidate.pt` and keeps the
previous promoted checkpoint intact. Advanced ablations can disable balancing or refitting
with `--no-balanced-sampling` and `--no-refit-train-validation`.

The split manifest is persisted at `reports/split-manifest.json`; evaluation always uses
that manifest rather than creating a new split.

## Tune temporal locking

The classifier remains a transparent single-frame MLP. A separate temporal policy decides
when enough protocol-aware evidence exists to seal the prediction. The current EMA rule is
always retained as the baseline; tuning compares it with a landmark-stability gate and a
small NumPy HMM-style filter with `READY_FIST`, `TRANSITION`, and three committed gesture
states.

Generate participant-disjoint out-of-fold predictions and freeze the fastest policy that
does not regress accuracy:

```bash
uv run --locked rps-tune-temporal tune
```

This trains seven fixed 32-epoch fold models, searches the deterministic policy grid, and
writes a report under `reports/`. The frozen result is stored at
`models/temporal_policy-candidate.json`; if nothing passes, its status is
`baseline_retained`, and confirmation refuses it. The demo never loads an unpromoted result.

Collect two genuinely new participants after tuning, with the same 20 repetitions per
class protocol, and review every new trajectory before confirmation. Confirmation requires
at least 20 included trajectories per class and participant, so replace excluded captures:

```bash
uv run --locked rps-capture --participant P08 --repetitions 20
uv run --locked rps-capture --participant P09 --repetitions 20
uv run --locked rps-review-data --participant P08
uv run --locked rps-review-data --participant P09
uv run --locked rps-tune-temporal confirm --participants P08 P09
```

Confirmation compares baseline and candidate on the same fresh trajectories. Promotion
requires at least 91.4% accuracy, no extra errors, at least 90% recall per class, a median
improvement of at least 25 ms, non-worse p90 latency, and under 0.1 ms p95 policy overhead.
Only a passing candidate is written to `models/temporal_policy.json` with the matching model
fingerprint and checkpoint hash. A missing or mismatched artifact makes the demo warn and use
the immutable baseline.

Evaluate either policy without changing promotion state:

```bash
uv run --locked rps-evaluate --split test
uv run --locked rps-evaluate --split test \
  --temporal-policy models/temporal_policy-candidate.json
```

Before confirmation, exercise asynchronous camera timing with the candidate in shadow mode;
the active policy still controls the round while paired decisions are printed:

```bash
uv run --locked rps-demo \
  --shadow-temporal-policy models/temporal_policy-candidate.json
```

## Development

```bash
uv run --locked ruff check .
uv run --locked pytest
```

Important modules:

- `features.py` owns the versioned landmark normalization used everywhere.
- `game.py` is a deterministic, camera-independent timed state machine.
- `temporal.py` owns replayable baseline, stability-gate, and HMM lock policies.
- `tracking.py` keeps only the newest asynchronous MediaPipe result.
- `renderer.py` draws the game-first booth view and alternate activation visualization.
- `audio.py` generates and asynchronously plays dependency-free local WAV cues.
- `data.py` owns the landmark-only format and participant-disjoint dataset construction.

Generated data, reports, downloaded assets, and local device benchmark state are ignored.
The small promoted PyTorch checkpoint is intended to be committed for reliable offline
booth operation.
