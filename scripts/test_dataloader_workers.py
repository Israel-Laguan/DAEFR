#!/usr/bin/env python3
"""
Diagnostic script to test DataLoader worker configurations
and determine safe num_workers for the system.
"""

import os
import sys
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import time
import argparse


class DummyDataset(Dataset):
    """Minimal dataset for testing DataLoader workers."""
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate light CPU work
        x = torch.randn(3, 512, 512)
        return x


def get_system_info():
    """Gather system capabilities."""
    info = {
        'cpu_count': os.cpu_count(),
        'usable_cpus': 0,
        'gpu_count': 0,
        'gpu_names': [],
        'thread_limit': 'unknown',
    }
    
    # Get usable CPUs (respects cgroup limits)
    try:
        info['usable_cpus'] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info['usable_cpus'] = info['cpu_count']
    
    # GPU info
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        for i in range(info['gpu_count']):
            info['gpu_names'].append(torch.cuda.get_device_name(i))
    
    # Check thread limits (common in containers)
    try:
        with open('/proc/sys/kernel/threads-max', 'r') as f:
            info['thread_limit'] = f.read().strip()
    except:
        pass
    
    # Check max user processes (ulimit -u)
    try:
        import resource
        info['max_user_processes'] = resource.getrlimit(resource.RLIMIT_NPROC)[0]
    except:
        info['max_user_processes'] = 'unknown'
    
    return info


def test_workers(num_workers, batch_size=4, num_batches=10, timeout=30):
    """Test if a specific num_workers configuration works."""
    dataset = DummyDataset(size=100)
    
    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            # Touch the data to ensure it's loaded
            _ = batch.sum().item()
        
        elapsed = time.time() - start
        return True, elapsed
        
    except RuntimeError as e:
        if "can't start new thread" in str(e):
            return False, 0.0
        raise
    except Exception as e:
        return False, 0.0


def estimate_safe_workers(system_info, num_gpus=1):
    """Estimate safe num_workers based on system capabilities."""
    usable = system_info['usable_cpus']
    
    # Conservative formula: usable cores / (GPUs * 3 loaders) - headroom
    # Each GPU process has train/val/test loaders
    num_loaders = 3
    total_processes = num_gpus * num_loaders
    
    # Reserve 2 cores for system/main process
    available = max(1, usable - 2)
    
    # Distribute across all loader instances
    per_loader = max(1, available // total_processes)
    
    # Cap at reasonable max (more workers = diminishing returns + thread overhead)
    safe = min(per_loader, 8)
    
    return {
        'usable_cpus': usable,
        'num_gpus': num_gpus,
        'num_loaders': num_loaders,
        'total_loader_instances': total_processes,
        'cores_per_loader_instance': per_loader,
        'recommended_workers': safe
    }


def main():
    parser = argparse.ArgumentParser(description='Test DataLoader worker configurations')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs for DDP training')
    parser.add_argument('--max-workers', type=int, default=16, help='Max workers to test')
    parser.add_argument('--quick', action='store_true', help='Quick test only recommended value')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DataLoader Worker Configuration Test")
    print("=" * 60)
    
    # System info
    info = get_system_info()
    print("\n[SYSTEM INFO]")
    print(f"  CPU cores (total):     {info['cpu_count']}")
    print(f"  CPU cores (usable):    {info['usable_cpus']}")
    print(f"  GPUs available:        {info['gpu_count']}")
    for i, name in enumerate(info['gpu_names']):
        print(f"    GPU {i}: {name}")
    print(f"  Thread limit:          {info['thread_limit']}")
    print(f"  Max user processes:    {info['max_user_processes']}")
    
    # Estimates
    print(f"\n[ESTIMATES for {args.gpus} GPU(s)]")
    estimates = estimate_safe_workers(info, args.gpus)
    print(f"  Usable CPU cores:      {estimates['usable_cpus']}")
    print(f"  DataLoader instances:  {estimates['total_loader_instances']} ({args.gpus} GPUs × 3 loaders)")
    print(f"  Cores per instance:    {estimates['cores_per_loader_instance']}")
    print(f"  RECOMMENDED workers:   {estimates['recommended_workers']}")
    
    if args.quick:
        workers_to_test = [estimates['recommended_workers']]
    else:
        # Test range: 0, 1, 2, 4, 8, and up to max-workers
        workers_to_test = sorted(set([0, 1, 2, 4, 8] + 
                                     list(range(4, min(args.max_workers + 1, 32), 4))))
    
    print(f"\n[TESTING CONFIGURATIONS]")
    print(f"{'Workers':>8} | {'Status':>10} | {'Time (s)':>10} | {'Notes'}")
    print("-" * 60)
    
    working_configs = []
    
    for nw in workers_to_test:
        if nw > args.max_workers:
            continue
            
        status, elapsed = test_workers(nw)
        
        if status:
            working_configs.append((nw, elapsed))
            note = "✓ OK"
            if nw == estimates['recommended_workers']:
                note += " <- recommended"
        else:
            note = "✗ FAILED - thread limit"
        
        time_str = f"{elapsed:.3f}" if status else "N/A"
        status_str = "OK" if status else "FAIL"
        
        print(f"{nw:>8} | {status_str:>10} | {time_str:>10} | {note}")
    
    # Final recommendation
    print("\n[RECOMMENDATION]")
    if working_configs:
        # Find best balance: lowest workers with good performance
        best = min(working_configs, key=lambda x: (x[0], -1/x[1] if x[1] > 0 else 0))
        print(f"  Safest configuration: num_workers = {best[0]}")
        print(f"\n  Update configs/DAEFR.yaml:")
        print(f"    num_workers: {best[0]}")
        
        # Also suggest command-line override
        print(f"\n  Or use command-line override:")
        print(f"    python main_DAEFR.py --base configs/DAEFR.yaml --train \\")
        print(f"        data.params.num_workers={best[0]}")
    else:
        print("  WARNING: No working configurations found!")
        print("  Try: num_workers: 0 (single-threaded, slower but reliable)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
