import random
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

@dataclass
class ProcessInfo:
    pid: int
    name: str
    gpu_memory: int  # in bytes
    gpu_util: int    # percentage

@dataclass
class GPUInfo:
    index: int
    name: str
    utilization: int  # percentage
    memory_total: int # bytes
    memory_used: int  # bytes
    temperature: int  # celsius
    power_draw: float # Watts
    power_limit: float # Watts
    fan_speed: int    # percentage
    cuda_cores: int   # total cores
    sm_count: int     # SM count
    clock_graphics: int # MHz
    clock_mem: int      # MHz
    util_history: List[int]
    mem_history: List[int]
    clock_history: List[int]
    temp_history: List[int]
    processes: List[ProcessInfo]

def get_cores_per_sm(major, minor):
    # Mapping of Compute Capability to CUDA Cores per SM
    # https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    mapping = {
        (2, 0): 32,
        (2, 1): 48,
        (3, 0): 192,
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,
        (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,
        (9, 0): 128,
    }
    return mapping.get((major, minor), 128) # Default to 128 for unknown newer ones

class GPUDataProvider:
    def __init__(self):
        self.initialized = False
        self.history_size = 50
        self.history = {} # index -> {'util': [], 'mem': [], 'clock': [], 'temp': []}
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.initialized = True
            except Exception:
                self.initialized = False

    def get_gpu_data(self) -> List[GPUInfo]:
        if not self.initialized:
            return self._get_mock_data()
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                try:
                    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                except:
                    power_limit = 0.0
                
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan = 0

                try:
                    sm_count = pynvml.nvmlDeviceGetMultiprocessorCount(handle)
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    cores_per_sm = get_cores_per_sm(major, minor)
                    cuda_cores = sm_count * cores_per_sm
                except:
                    sm_count = 0
                    cuda_cores = 0

                try:
                    clock_gpu = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    clock_gpu = 0
                    clock_mem = 0

                # Update History
                if i not in self.history:
                    self.history[i] = {'util': [], 'mem': [], 'clock': [], 'temp': []}
                
                h = self.history[i]
                h['util'].append(util.gpu)
                h['mem'].append(int((mem.used / mem.total) * 100) if mem.total > 0 else 0)
                h['clock'].append(clock_gpu)
                h['temp'].append(temp)
                
                for key in h:
                    if len(h[key]) > self.history_size:
                        h[key].pop(0)

                # Processes
                procs = []
                try:
                    nv_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in nv_procs:
                        # Attempt to get process name
                        try:
                            import psutil
                            proc_name = psutil.Process(p.pid).name()
                        except:
                            proc_name = "Unknown"
                        procs.append(ProcessInfo(p.pid, proc_name, p.usedGpuMemory, 0)) # Utilization per process is harder to get via NVML directly
                except:
                    pass

                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    utilization=util.gpu,
                    memory_total=mem.total,
                    memory_used=mem.used,
                    temperature=temp,
                    power_draw=power,
                    power_limit=power_limit,
                    fan_speed=fan,
                    cuda_cores=cuda_cores,
                    sm_count=sm_count,
                    clock_graphics=clock_gpu,
                    clock_mem=clock_mem,
                    util_history=list(h['util']),
                    mem_history=list(h['mem']),
                    clock_history=list(h['clock']),
                    temp_history=list(h['temp']),
                    processes=procs
                ))
            return gpus
        except Exception:
            return self._get_mock_data()

    def _get_mock_data(self) -> List[GPUInfo]:
        # Return some fake data for testing on non-NVIDIA systems
        gpus = []
        for i in range(2):
            gpus.append(GPUInfo(
                index=i,
                name=f"NVIDIA GeForce RTX 4090 (Mock {i})",
                utilization=random.randint(0, 100),
                memory_total=24 * 1024 * 1024 * 1024,
                memory_used=random.randint(2, 20) * 1024 * 1024 * 1024,
                temperature=random.randint(30, 85),
                power_draw=random.uniform(50.0, 450.0),
                power_limit=450.0,
                fan_speed=random.randint(0, 100),
                cuda_cores=16384,
                sm_count=128,
                clock_graphics=random.randint(1500, 2500),
                clock_mem=random.randint(5000, 10000),
                util_history=[random.randint(0,100) for _ in range(20)],
                mem_history=[random.randint(0,100) for _ in range(20)],
                clock_history=[random.randint(1500,2500) for _ in range(20)],
                temp_history=[random.randint(30,85) for _ in range(20)],
                processes=[
                    ProcessInfo(1234, "python (train.py)", 8 * 1024 * 1024 * 1024, 45),
                    ProcessInfo(5678, "llama-cpp", 4 * 1024 * 1024 * 1024, 20),
                ]
            ))
        return gpus

    def __del__(self):
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
