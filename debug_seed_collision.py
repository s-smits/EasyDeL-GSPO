#!/usr/bin/env python3
"""Debug script to check for PRNG seed collisions in GRPO training."""

def generate_seeds_like_grpo(cur_step_int=0, process_index=0, chunk_idx=0, num_returns=8):
    """Generate seeds exactly like GRPO trainer does."""
    # Base per-chunk seed calculation
    per_chunk_seed = int((cur_step_int * 131071 + 4099 * abs(process_index) + chunk_idx) % (2**31 - 1))
    per_chunk_seed = max(1, per_chunk_seed)
    
    print(f"Step {cur_step_int}, Process {process_index}, Chunk {chunk_idx}:")
    print(f"  per_chunk_seed: {per_chunk_seed}")
    
    # Generate seeds for each return
    seeds = []
    for ri in range(num_returns):
        seed_ri = int((per_chunk_seed + (ri + 1) * 104729) % (2**31 - 1))
        seed_ri = max(1, seed_ri)
        seeds.append(seed_ri)
        print(f"  return {ri}: seed={seed_ri}")
    
    return per_chunk_seed, seeds

def check_for_collisions():
    """Check if seeds collide across different configurations."""
    all_seeds = set()
    collision_count = 0
    
    # Test various configurations
    for step in range(3):  # First few steps
        for chunk in range(2):  # First few chunks
            base_seed, return_seeds = generate_seeds_like_grpo(
                cur_step_int=step, 
                process_index=0, 
                chunk_idx=chunk,
                num_returns=8
            )
            
            # Check for collisions
            for seed in return_seeds:
                if seed in all_seeds:
                    print(f"COLLISION DETECTED: seed {seed}")
                    collision_count += 1
                else:
                    all_seeds.add(seed)
            
            print()
    
    print(f"Total unique seeds: {len(all_seeds)}")
    print(f"Total collisions: {collision_count}")
    
    # Also test if consecutive seeds are too similar
    test_seeds = list(all_seeds)[:10]
    print(f"\nFirst 10 seeds: {test_seeds}")
    
    # Check distance between consecutive seeds
    for i in range(len(test_seeds) - 1):
        diff = abs(test_seeds[i] - test_seeds[i+1])
        print(f"  Seed {i} vs {i+1}: diff = {diff}")

if __name__ == "__main__":
    check_for_collisions()