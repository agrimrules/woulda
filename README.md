# Woulda: NVIDIA GPU Performance Monitor

A modern Terminal UI (TUI) for visualizing NVIDIA GPU performance, built with [Textual](https://textual.textualize.io/).

## Features
- **Live Dashboard**: Real-time updates of GPU utilization, memory, power, and temperature.
- **CUDA Core Insights**: Visualizes active CUDA cores based on GPU architecture (Compute Capability).
- **Process Monitoring**: Tracks processes running on each GPU with their memory consumption.
- **Cross-Platform Mock Mode**: Automatically enters mock mode if no NVIDIA GPU is detected, allowing for TUI development on any machine.

## Installation

1. Ensure you have Python 3.8+ installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install NVIDIA drivers and NVML (usually included with CUDA/Drivers).

## Usage

Run the application:
```bash
python main.py
```

### Controls
- `q`: Quit the application.
- `r`: Force manual refresh.

## Project Structure
- `main.py`: The Textual TUI application and UI components.
- `gpu_provider.py`: Data abstraction layer with NVML support and Mock fallback.
- `requirements.txt`: Python package dependencies.
