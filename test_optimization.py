#!/usr/bin/env python3
"""
Test script to verify PID optimization functionality.
This runs a quick optimization test with very short simulation durations.
"""

import sys
sys.path.insert(0, '.')

from main import simulate_pid, optimize_pid

if __name__ == "__main__":
    print("=" * 70)
    print("PID OPTIMIZATION TEST")
    print("=" * 70)
    
    # Test 1: Single simulation run
    print("\n1. Testing single simulation run (loss calculation)...")
    loss = simulate_pid(kp=50.0, ki=0.1, kd=10.0, duration=1.0)
    print(f"   Loss for Kp=50, Ki=0.1, Kd=10: {loss:.4f}")
    
    # Test 2: Brief optimization
    print("\n2. Running brief PID optimization (will take ~1-2 minutes)...")
    print("   Initial guess: Kp=50.0, Ki=0.0, Kd=10.0")
    print("   Each evaluation runs 1 second of simulation")
    print("   Optimization algorithm: L-BFGS-B with ~10-20 evaluations\n")
    
    kp_opt, ki_opt, kd_opt = optimize_pid(
        initial_kp=50.0,
        initial_ki=0.0,
        initial_kd=10.0,
        duration=10.0  # 1 second per evaluation for quick test
    )
    
    print("\n3. Optimization results:")
    print(f"   Optimal Kp: {kp_opt:.3f}")
    print(f"   Optimal Ki: {ki_opt:.3f}")
    print(f"   Optimal Kd: {kd_opt:.3f}")
    
    print("\n" + "=" * 70)
    print("To use optimized parameters in main.py:")
    print("  1. Open main.py")
    print("  2. Set ENABLE_OPTIMIZATION = True")
    print("  3. Run: python main.py")
    print("  The optimization will run first, then load values into sliders")
    print("=" * 70 + "\n")
