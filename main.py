import humanize
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, ProgressBar, DataTable, Label, Sparkline
from textual.reactive import reactive
from textual.timer import Timer

from gpu_provider import GPUDataProvider, GPUInfo

class GPUChart(Static):
    """A widget to display a sparkline and a label."""
    def __init__(self, label: str, data: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.label_text = label
        self.data = data or [0] * 20

    def compose(self) -> ComposeResult:
        yield Label(self.label_text, classes="chart-label")
        yield Sparkline(self.data, id="sparkline")

    def update_data(self, data: List[int]):
        self.query_one(Sparkline).data = data

class GPUCard(Static):
    """A widget to display information about a single GPU."""
    
    def __init__(self, gpu_info: GPUInfo, **kwargs):
        super().__init__(**kwargs)
        self.gpu_info = gpu_info

    def compose(self) -> ComposeResult:
        with Vertical(classes="card-container"):
            yield Label(f"[bold cyan]GPU {self.gpu_info.index}: {self.gpu_info.name}[/bold cyan]")
            
            with Horizontal(classes="metric-row"):
                yield Label("Utilization: ", classes="metric-label")
                yield ProgressBar(total=100, show_percentage=True, id=f"util-{self.gpu_info.index}")
            
            with Horizontal(classes="metric-row"):
                yield Label("Memory:      ", classes="metric-label")
                yield ProgressBar(total=100, show_percentage=True, id=f"mem-{self.gpu_info.index}")
            
            with Horizontal(classes="info-row"):
                yield Label(id=f"temp-{self.gpu_info.index}", classes="info-item")
                yield Label(id=f"power-{self.gpu_info.index}", classes="info-item")
                yield Label(id=f"fan-{self.gpu_info.index}", classes="info-item")
            
            with Horizontal(classes="info-row"):
                yield Label(id=f"cores-{self.gpu_info.index}", classes="info-item-wide")
                yield Label(id=f"sms-{self.gpu_info.index}", classes="info-item")
            
            with Horizontal(classes="chart-row"):
                yield GPUChart("Utilization %", id=f"util-chart-{self.gpu_info.index}")
                yield GPUChart("Temperature °C", id=f"temp-chart-{self.gpu_info.index}")
                yield GPUChart("Graphics Clock", id=f"clock-chart-{self.gpu_info.index}")

    def update_info(self, info: GPUInfo):
        self.gpu_info = info
        self.query_one(f"#util-{info.index}", ProgressBar).progress = info.utilization
        mem_pct = (info.memory_used / info.memory_total) * 100 if info.memory_total > 0 else 0
        self.query_one(f"#mem-{info.index}", ProgressBar).progress = mem_pct
        
        self.query_one(f"#temp-{info.index}", Label).update(f"Temp: [bold yellow]{info.temperature}°C[/bold yellow]")
        self.query_one(f"#power-{info.index}", Label).update(f"Power: [bold green]{info.power_draw:.1f}W / {info.power_limit:.1f}W[/bold green]")
        self.query_one(f"#fan-{info.index}", Label).update(f"Fan: [bold white]{info.fan_speed}%[/bold white]")
        
        cores_active = int((info.utilization / 100.0) * info.cuda_cores)
        self.query_one(f"#cores-{info.index}", Label).update(f"Cores: [bold cyan]{cores_active:,} / {info.cuda_cores:,} active[/bold cyan]")
        self.query_one(f"#sms-{info.index}", Label).update(f"SMs: [bold cyan]{info.sm_count}[/bold cyan]")

        # Update Charts
        self.query_one(f"#util-chart-{info.index}", GPUChart).update_data(info.util_history)
        self.query_one(f"#temp-chart-{info.index}", GPUChart).update_data(info.temp_history)
        self.query_one(f"#clock-chart-{info.index}", GPUChart).update_data(info.clock_history)

    def on_mount(self) -> None:
        self.update_info(self.gpu_info)

class GPUVisualizerApp(App):
    """A Textual app to visualize NVIDIA GPU performance."""
    
    CSS = """
    Screen {
        background: #1a1b26;
    }
    
    .card-container {
        border: solid #3b4261;
        background: #24283b;
        padding: 1;
        margin: 1;
        height: auto;
    }
    
    .metric-row {
        height: 1;
        margin-top: 1;
    }
    
    .metric-label {
        width: 15;
    }
    
    .info-row {
        margin-top: 1;
        height: 1;
    }
    
    .info-item {
        margin-right: 4;
    }

    .info-item-wide {
        width: 30;
        margin-right: 4;
    }

    .chart-row {
        height: 4;
        margin-top: 1;
    }

    GPUChart {
        width: 1fr;
        height: 4;
        margin: 0 1;
        border: solid #3b4261;
        padding: 0 1;
    }

    .chart-label {
        color: #7aa2f7;
        text-align: center;
        width: 100%;
    }
    
    ProgressBar {
        width: 1fr;
    }
    
    ProgressBar > .bar--bar {
        color: #7aa2f7;
    }
    
    ProgressBar > .bar--complete {
        color: #73daca;
    }

    #process-list {
        height: 1fr;
        border: solid #3b4261;
        background: #24283b;
        margin: 1;
    }
    
    DataTable {
        background: transparent;
    }
    """

    TITLE = "NVIDIA GPU Performance Monitor"
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

    def __init__(self):
        super().__init__()
        self.provider = GPUDataProvider()
        self.gpu_widgets = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical():
                yield Static("[bold underline]GPU Status[/bold underline]", id="gpu-header")
                self.gpu_grid = Grid(id="gpu-grid")
                yield self.gpu_grid
                
                yield Static("[bold underline]Running Processes[/bold underline]", id="process-header")
                yield DataTable(id="process-list")
        yield Footer()

    def on_mount(self) -> None:
        self.table = self.query_one(DataTable)
        self.table.add_columns("GPU", "PID", "Name", "Memory Used")
        self.table.cursor_type = "row"
        
        # Initial data load
        self.update_data()
        
        # Set up a timer for auto-refresh
        self.set_interval(2.0, self.update_data)

    def update_data(self) -> None:
        gpus = self.provider.get_gpu_data()
        
        # Update GPU cards
        grid = self.query_one("#gpu-grid")
        for gpu in gpus:
            if gpu.index not in self.gpu_widgets:
                card = GPUCard(gpu, id=f"gpu-card-{gpu.index}")
                self.gpu_widgets[gpu.index] = card
                grid.mount(card)
            else:
                self.gpu_widgets[gpu.index].update_info(gpu)
        
        # Update Process Table
        self.table.clear()
        for gpu in gpus:
            for proc in gpu.processes:
                mem_str = humanize.naturalsize(proc.gpu_memory, binary=True)
                self.table.add_row(
                    str(gpu.index),
                    str(proc.pid),
                    proc.name,
                    mem_str
                )

if __name__ == "__main__":
    app = GPUVisualizerApp()
    app.run()
