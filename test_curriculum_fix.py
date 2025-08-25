#!/usr/bin/env python3
"""Test script to verify curriculum learning fix works properly."""

import os
import sys
import subprocess
import time

def test_curriculum_training():
    """Run a quick test of curriculum training with minimal settings."""
    
    print("=" * 60)
    print("Testing Curriculum Learning Fix")
    print("=" * 60)
    
    # Test with very small settings for quick validation
    cmd = [
        "python", "-m", "easydel.scripts.finetune.gsm8k_math_gfspo",
        "--repo_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset", "math-ds",
        "--dataset_use_pct", "10",  # Use only 10% of data for quick test
        "--curriculum_math", "true",
        "--num_train_epochs", "1",  # 1 epoch per level for quick test
        "--learning_rate", "5e-6",
        "--optimizer", "adamw",
        "--total_batch_size", "4",
        "--mini_batch_size", "1",
        "--force_tensor_parallel", "4",
        "--force_data_parallel", "8",
        "--num_return_sequences", "4",
        "--gfpo_group_size", "4",
        "--gfpo_retain_count", "2",
        "--save_directory", "/tmp/curriculum_test",
        "--save_steps", "10",
        "--evaluation_steps", "10",
        "--do_eval", "false",  # Skip eval for speed
        "--do_last_save", "false",  # Skip final save for speed
        "--max_training_steps", "20",  # Limit total steps for quick test
        "--gradient_accumulation_steps", "1",
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Run the training with a timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        curriculum_levels_seen = []
        shutdown_detected = False
        error_detected = False
        training_started = False
        
        # Monitor output
        start_time = time.time()
        timeout = 300  # 5 minute timeout for test
        
        while True:
            if time.time() - start_time > timeout:
                print(f"\nTimeout reached ({timeout}s), terminating test...")
                process.terminate()
                break
                
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
                
            print(line.rstrip())
            
            # Check for curriculum level progress
            if "Curriculum Stage" in line:
                training_started = True
                level = line.split(":")[-1].strip()
                curriculum_levels_seen.append(level)
                print(f"\n>>> Detected curriculum level: {level}")
                
            # Check for successful completion of levels
            if "Completed training for Level" in line:
                print(f"\n>>> Level completed successfully!")
                
            # Check for shutdown issues
            if "Grain pool is exiting" in line and not training_started:
                shutdown_detected = True
                print("\n>>> WARNING: Early shutdown detected!")
                
            # Check for errors
            if "ERROR" in line or "Exception" in line or "Traceback" in line:
                error_detected = True
                print(f"\n>>> ERROR detected: {line.rstrip()}")
                
            # Check if curriculum completed
            if "Curriculum learning completed!" in line:
                print("\n>>> SUCCESS: Curriculum learning completed!")
                break
                
        process.wait()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print("-" * 60)
        print(f"Curriculum levels seen: {curriculum_levels_seen}")
        print(f"Training started: {training_started}")
        print(f"Early shutdown detected: {shutdown_detected}")
        print(f"Errors detected: {error_detected}")
        print(f"Exit code: {process.returncode}")
        
        if len(curriculum_levels_seen) > 1 and not shutdown_detected and not error_detected:
            print("\n✅ TEST PASSED: Curriculum learning progressed through multiple levels")
            return True
        else:
            print("\n❌ TEST FAILED: Issues detected with curriculum learning")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        return False
        
if __name__ == "__main__":
    success = test_curriculum_training()
    sys.exit(0 if success else 1)
