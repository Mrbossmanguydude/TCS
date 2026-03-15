# Traffic Control System (TCS)

Traffic Control System (TCS) is a Pygame-based simulation and training environment for centralised traffic control using PPO.

The project provides:
- A multi-screen GUI for setup, training, evaluation, replay, baseline demo, options, and controls.
- Curriculum-based training across phases and map levels.
- Network slot management (save/load/delete/rename).
- Episode replay slot management with playback.
- Runtime logging to SQLite and JSON exports.

## Current Versioning Style

Commits use a version label inside commit messages, for example:
- `VERSION 2.3`
- `VERSION 3.4`
- `VERSION 3.5`

This README includes commands to find and run specific versions.

## Requirements

- Python 3.11+ (recommended)
- `pygame-ce`
- `torch`
- `numpy`

## 1) Download / Clone

### Option A: Clone with Git

```powershell
git clone <your-repo-url>
cd TCS
```

### Option B: Download ZIP

1. Download the repository ZIP from your Git host.
2. Extract it.
3. Open a terminal in the extracted `TCS` folder.

## 2) Install Dependencies

### Windows PowerShell (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install pygame-ce torch numpy
```

If you already have a compatible Python environment, you can skip virtual environment creation.

## 3) Run the Application

From the project root:

```powershell
python src\gui\gui_main.py
```

OR

Run out of src/gui_main.py.

Main menu screens:
- Train
- Evaluate
- Demo
- Replays
- Setup
- Options
- Controls

## 4) Data Produced at Runtime

The app writes runtime files under `data/`, including:
- `data/metrics/tcs.db` (SQLite run/episode/metric logs)
- `data/models/` (network checkpoints and metadata)
- `data/replays/` (episode replay slot payloads)
- `data/screenshots/` (manual screenshots)
- `data/logs/` (config snapshots and exports)

## 5) How to Look Up Commit Versions

### Show recent commit history

```powershell
git log --oneline --decorate --graph -n 30
```

### Find a specific version by message text

```powershell
git log --oneline --grep "VERSION 3.5"
git log --oneline --grep "VERSION 3.4"
git log --oneline --grep "VERSION 2.3"
```

### View full commit details

```powershell
git show <commit-hash>
```

### View commit files changed summary

```powershell
git show --stat <commit-hash>
```

## 6) Run an Older Commit Version (without changing history)

1. Save or commit your current work first.
2. Switch to the commit in detached HEAD mode:

```powershell
git switch --detach <commit-hash>
```

3. Run that version:

```powershell
python src\gui\gui_main.py
```

4. Return to your normal branch when done:

```powershell
git switch main
```

If your main branch has a different name, use:

```powershell
git branch
```

and switch to the correct branch.

## 7) Useful Git Checks

Current branch:

```powershell
git branch --show-current
```

Current commit:

```powershell
git rev-parse --short HEAD
```

Working tree status:

```powershell
git status
```

## 8) High-Level Code Layout

- `src/gui/`
  - `gui_main.py`: application entrypoint and top-level state routing
  - `setup_screen.py`: setup and map preview controls
  - `train_screen.py`: training setup and during-training runtime
  - `evaluation_screen.py`: evaluation runs with loaded network
  - `baseline_demo_screen.py`: non-training demonstration runs
  - `replay_screen.py`: network and episode replay browser
  - `options_screen.py`: general and advanced configuration
  - `controls_screen.py`: per-screen controls reference
  - `ui_offsets.py`: layout offsets and sizing controls

- `src/utils/`
  - `run_init.py`: runtime bootstrap, config, database lifecycle
  - `ppo_controller.py`: PPO model, action selection, GAE, update
  - `train_backend_helpers.py`: training-step support helpers
  - `controller_prep.py`: observation and controller prep support
  - `map_generation.py`: procedural map generation
  - `network_slots.py`: fixed network slot storage helpers
  - `replay.py`: episode replay slot persistence helpers
  - `train_types.py`: training datatypes
  - `hold_repeat.py`: hold-to-repeat +/- input behaviour

## 9) Notes

- This project is currently tuned for local desktop use.
- If you are testing historical versions, behaviour and available screens may differ by commit.
