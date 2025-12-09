#!/usr/bin/env python3
"""
Benchmark script for InLineHoloController performance on Raspberry Pi 5

This script tests different optimization combinations and measures:
- Capture FPS
- Processing FPS  
- Average processing time
- Dropped frames

Usage:
    python benchmark_hologram_processing.py [--duration SECONDS] [--roi-size SIZE]

Example:
    python benchmark_hologram_processing.py --duration 30 --roi-size 256
"""

import argparse
import time
import json
from datetime import datetime
import numpy as np

# Test configurations to benchmark
BENCHMARK_CONFIGS = [
    {
        "name": "Baseline (NumPy FFT, float64, threading)",
        "params": {
            "use_scipy_fft": False,
            "use_float32": False,
            "use_multiprocessing": False,
            "enable_benchmarking": True,
            "update_freq": 30.0,
        }
    },
    {
        "name": "Float32 optimization",
        "params": {
            "use_scipy_fft": False,
            "use_float32": True,
            "use_multiprocessing": False,
            "enable_benchmarking": True,
            "update_freq": 30.0,
        }
    },
    {
        "name": "SciPy FFT (4 workers)",
        "params": {
            "use_scipy_fft": True,
            "fft_workers": 4,
            "use_float32": True,
            "use_multiprocessing": False,
            "enable_benchmarking": True,
            "update_freq": 30.0,
        }
    },
    {
        "name": "SciPy FFT (2 workers)",
        "params": {
            "use_scipy_fft": True,
            "fft_workers": 2,
            "use_float32": True,
            "use_multiprocessing": False,
            "enable_benchmarking": True,
            "update_freq": 30.0,
        }
    },
    {
        "name": "Multiprocessing mode",
        "params": {
            "use_scipy_fft": True,
            "fft_workers": 4,
            "use_float32": True,
            "use_multiprocessing": True,
            "enable_benchmarking": True,
            "update_freq": 30.0,
        }
    },
]


def run_benchmark_via_api(base_url, config, duration, roi_size, dz):
    """
    Run a single benchmark configuration via REST API
    
    Args:
        base_url: Base URL of ImSwitch REST API (e.g., "http://localhost:8001")
        config: Configuration dict with name and params
        duration: How long to run benchmark in seconds
        roi_size: ROI size in pixels
        dz: Propagation distance in meters
    
    Returns:
        Dict with benchmark results
    """
    import requests
    
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    
    # Set parameters
    params = config['params'].copy()
    params['roi_size'] = roi_size
    params['dz'] = dz
    
    try:
        response = requests.post(
            f"{base_url}/holocontroller/set_parameters_inlineholo",
            json=params
        )
        response.raise_for_status()
        print(f"✓ Parameters set")
    except Exception as e:
        print(f"✗ Failed to set parameters: {e}")
        return None
    
    # Start processing
    try:
        response = requests.get(f"{base_url}/holocontroller/start_processing_inlineholo")
        response.raise_for_status()
        print(f"✓ Processing started")
    except Exception as e:
        print(f"✗ Failed to start processing: {e}")
        return None
    
    # Wait for warmup
    print(f"  Warming up (5s)...")
    time.sleep(5)
    
    # Collect metrics during test period
    print(f"  Running benchmark ({duration}s)...")
    start_time = time.time()
    samples = []
    
    while time.time() - start_time < duration:
        try:
            response = requests.get(f"{base_url}/holocontroller/get_state_inlineholo")
            response.raise_for_status()
            state = response.json()
            samples.append({
                'timestamp': time.time(),
                'capture_fps': state.get('capture_fps', 0),
                'processing_fps': state.get('processing_fps', 0),
                'avg_process_time': state.get('avg_process_time', 0),
                'dropped_frames': state.get('dropped_frames', 0),
                'processed_count': state.get('processed_count', 0),
            })
        except Exception as e:
            print(f"  Warning: Failed to get state: {e}")
        
        time.sleep(1.0)
    
    # Stop processing
    try:
        response = requests.get(f"{base_url}/holocontroller/stop_processing_inlineholo")
        response.raise_for_status()
        print(f"✓ Processing stopped")
    except Exception as e:
        print(f"✗ Failed to stop processing: {e}")
    
    # Calculate statistics
    if not samples:
        print(f"✗ No samples collected")
        return None
    
    capture_fps_values = [s['capture_fps'] for s in samples if s['capture_fps'] > 0]
    processing_fps_values = [s['processing_fps'] for s in samples if s['processing_fps'] > 0]
    process_time_values = [s['avg_process_time'] for s in samples if s['avg_process_time'] > 0]
    
    results = {
        'config_name': config['name'],
        'params': params,
        'duration': duration,
        'samples_collected': len(samples),
        'capture_fps_mean': np.mean(capture_fps_values) if capture_fps_values else 0,
        'capture_fps_std': np.std(capture_fps_values) if capture_fps_values else 0,
        'processing_fps_mean': np.mean(processing_fps_values) if processing_fps_values else 0,
        'processing_fps_std': np.std(processing_fps_values) if processing_fps_values else 0,
        'avg_process_time_mean': np.mean(process_time_values) if process_time_values else 0,
        'avg_process_time_std': np.std(process_time_values) if process_time_values else 0,
        'total_dropped_frames': samples[-1]['dropped_frames'] if samples else 0,
        'total_processed_frames': samples[-1]['processed_count'] if samples else 0,
    }
    
    # Print results
    print(f"\n  Results:")
    print(f"    Capture FPS:     {results['capture_fps_mean']:.2f} ± {results['capture_fps_std']:.2f}")
    print(f"    Processing FPS:  {results['processing_fps_mean']:.2f} ± {results['processing_fps_std']:.2f}")
    print(f"    Avg Process Time: {results['avg_process_time_mean']*1000:.2f} ± {results['avg_process_time_std']*1000:.2f} ms")
    print(f"    Dropped Frames:  {results['total_dropped_frames']}")
    print(f"    Total Processed: {results['total_processed_frames']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark hologram processing performance")
    parser.add_argument("--base-url", default="http://localhost:8001",
                        help="Base URL of ImSwitch REST API")
    parser.add_argument("--duration", type=int, default=20,
                        help="Duration of each benchmark run in seconds")
    parser.add_argument("--roi-size", type=int, default=256,
                        help="ROI size in pixels")
    parser.add_argument("--dz", type=float, default=0.005,
                        help="Propagation distance in meters")
    parser.add_argument("--output", default=None,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Hologram Processing Benchmark")
    print(f"{'='*60}")
    print(f"Base URL: {args.base_url}")
    print(f"Duration per test: {args.duration}s")
    print(f"ROI size: {args.roi_size}x{args.roi_size}")
    print(f"Propagation distance: {args.dz}m")
    print(f"Configurations to test: {len(BENCHMARK_CONFIGS)}")
    
    # Run all benchmarks
    all_results = []
    
    for config in BENCHMARK_CONFIGS:
        result = run_benchmark_via_api(
            args.base_url,
            config,
            args.duration,
            args.roi_size,
            args.dz
        )
        
        if result:
            all_results.append(result)
        
        # Short pause between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    if not all_results:
        print("No successful benchmarks completed")
        return
    
    # Find baseline
    baseline = all_results[0]
    
    print(f"\n{'Configuration':<40} {'Process FPS':<15} {'Speedup':<10} {'Avg Time (ms)':<15}")
    print(f"{'-'*80}")
    
    for result in all_results:
        speedup = result['processing_fps_mean'] / baseline['processing_fps_mean'] if baseline['processing_fps_mean'] > 0 else 0
        print(
            f"{result['config_name']:<40} "
            f"{result['processing_fps_mean']:>6.2f} ± {result['processing_fps_std']:>5.2f}  "
            f"{speedup:>6.2f}x     "
            f"{result['avg_process_time_mean']*1000:>6.2f} ± {result['avg_process_time_std']*1000:>5.2f}"
        )
    
    # Save results
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_params': {
                'duration': args.duration,
                'roi_size': args.roi_size,
                'dz': args.dz,
            },
            'results': all_results,
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {args.output}")
    
    # Print recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    best_config = max(all_results, key=lambda x: x['processing_fps_mean'])
    speedup = best_config['processing_fps_mean'] / baseline['processing_fps_mean']
    
    print(f"\nBest configuration: {best_config['config_name']}")
    print(f"  Processing FPS: {best_config['processing_fps_mean']:.2f} fps")
    print(f"  Speedup: {speedup:.2f}x over baseline")
    print(f"  Avg processing time: {best_config['avg_process_time_mean']*1000:.2f} ms")
    
    if best_config['total_dropped_frames'] > 0:
        print(f"\n⚠ Warning: {best_config['total_dropped_frames']} frames dropped")
        print(f"  Consider reducing update_freq or ROI size")


if __name__ == "__main__":
    main()
