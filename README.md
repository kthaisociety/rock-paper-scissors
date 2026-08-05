# Mid-Gesture Rock-Paper-Scissors AI

An interactive student-fair demo that predicts a visitor's rock-paper-scissors move while
their fingers are still moving. MediaPipe tracks one hand, a small PyTorch MLP classifies
each partial pose, and the game commits to a counter-move before the final reveal. The
OpenCV display includes a live graph of the MLP's activations and weighted connections.

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

Add `--fullscreen` for the booth display. Press `q` or Escape to exit.

Without `models/gesture_mlp.pt`, the application intentionally opens in a prominent
`UNTRAINED - VISUALIZATION MODE`: random activations are shown, but gameplay and scoring
are disabled. This makes UI development possible without presenting random predictions as
a trained model.

Each trained round follows this timeline:

```text
closed-fist ready pose -> countdown -> GO
150 ms: continuous partial-pose predictions begin
200-450 ms: early confidence lock or forced deadline lock
650-950 ms: held final pose is scored
950 ms: the sealed AI move and result are revealed
```

The AI move is hidden after it locks so the visitor cannot change their move in response.

## Collect training trajectories

Collect data from 8-12 people; ten participants with 20 repetitions per class produces the
target 600 trajectories. Use a non-identifying participant alias:

```bash
uv run --locked rps-capture --participant P01 --repetitions 20 --camera 0
```

The capture UI randomizes prompts, prevents three identical prompts in a row, and repeats
trials without at least four tracked frames in both the early and final windows. Files are
written atomically beneath `data/landmarks/<participant>/<session>/` and contain:

- 21 x 3 landmarks and relative timestamps
- Handedness
- Prompted class
- Participant alias, session ID, trajectory ID, and capture configuration

No image or video encoding exists in the capture path. The complete data directory is
ignored by Git.

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

AdamW trains for at most 100 epochs with validation early stopping. A scalar temperature
is fitted after model selection, and 99th-percentile activation scales are stored for a
stable live visualization. Training writes a JSON report under `reports/` and promotes the
model to `models/gesture_mlp.pt` only when the held-out test metrics satisfy all booth
thresholds. Otherwise it writes `models/gesture_mlp-candidate.pt` and prints the failed
criteria.

The split manifest is persisted at `reports/split-manifest.json`; evaluation always uses
that manifest rather than creating a new split.

## Development

```bash
uv run --locked ruff check .
uv run --locked pytest
```

Important modules:

- `features.py` owns the versioned landmark normalization used everywhere.
- `game.py` is a deterministic, camera-independent timed state machine.
- `tracking.py` keeps only the newest asynchronous MediaPipe result.
- `renderer.py` draws feature groups, activations, weighted contributions, probabilities,
  timing, and game state.
- `data.py` owns the landmark-only format and participant-disjoint dataset construction.

Generated data, reports, downloaded assets, and local device benchmark state are ignored.
The small promoted PyTorch checkpoint is intended to be committed for reliable offline
booth operation.
