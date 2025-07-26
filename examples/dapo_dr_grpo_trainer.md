# DAPO and DR GRPO Training with EasyDeL: Advanced GRPO Variants

## Overview

This tutorial demonstrates how to use EasyDeL's advanced GRPO variants: **DAPO (Diversity-Aware Policy Optimization)** and **DR GRPO (GRPO Done Right)**. Both represent significant improvements over the original GRPO algorithm, addressing different limitations and achieving state-of-the-art results in mathematical reasoning tasks.

## Algorithm Comparison

| Algorithm | Key Innovation | Performance | Best For |
|-----------|---------------|-------------|----------|
| **GRPO** | Group-based advantage estimation | Baseline | General RL training |
| **DAPO** | Asymmetric clipping + Dynamic sampling + Token-level loss | 50 AIME 2024 points | Long reasoning chains |
| **DR GRPO** | Corrected normalization biases | 43.3 AIME 2024 points | Efficient training |

## DAPO (Diversity-Aware Policy Optimization)

DAPO addresses four critical limitations in GRPO:

### 1. Asymmetric Clipping (Clip-Higher)
- **Problem**: GRPO's symmetric clipping causes entropy collapse
- **Solution**: Different clipping ratios for exploration vs stability
- **Implementation**: `clip_ratio_low=0.2, clip_ratio_high=0.28`

### 2. Dynamic Sampling
- **Problem**: Gradient vanishing when all rewards are uniform
- **Solution**: Resample batches with insufficient diversity
- **Impact**: +8% accuracy improvement (largest component gain)

### 3. Token-Level Policy Gradient Loss
- **Problem**: Length bias disadvantages longer sequences
- **Solution**: Equal weighting across all tokens
- **Result**: Prevents dilution of learning signals

### 4. Overlong Reward Shaping
- **Problem**: Hard truncation creates reward noise
- **Solution**: Graduated soft punishment with buffer zone

## DR GRPO (GRPO Done Right)

DR GRPO corrects fundamental mathematical biases in GRPO:

### 1. Length Bias Elimination
- **Problem**: Sequence-level normalization under-penalizes longer incorrect responses
- **Solution**: Constant normalization factor instead of length division

### 2. Standard Deviation Bias Removal
- **Problem**: Normalizing by std(r) gives unequal weight to questions
- **Solution**: Treat all questions equally during optimization

## Basic Usage Examples

### DAPO Configuration

```python
import easydel as ed
import jax.numpy as jnp

# DAPO configuration with all improvements enabled
dapo_config = ed.DAPOConfig(
    # Basic training parameters
    save_directory="dapo-math-solver",
    num_train_epochs=3,
    total_batch_size=8,
    learning_rate=5e-7,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_return_sequences=4,
    
    # DAPO-specific: Asymmetric clipping
    clip_ratio_low=0.2,      # Standard lower bound
    clip_ratio_high=0.28,    # Higher upper bound for exploration
    
    # DAPO-specific: Dynamic sampling
    enable_dynamic_sampling=True,
    max_resample_attempts=3,
    min_accuracy_variance=0.1,
    
    # DAPO-specific: Token-level loss
    use_token_level_loss=True,
    
    # DAPO-specific: Overlong reward shaping
    enable_overlong_reward_shaping=True,
    overlong_buffer_length=4096,
    overlong_penalty_scale=0.1,
    
    # Typically beta=0.0 in DAPO (no KL penalty)
    beta=0.0,
    
    # Generation parameters for diversity
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)
```

### DR GRPO Configuration

```python
# DR GRPO configuration with bias corrections
dr_grpo_config = ed.DRGRPOConfig(
    # Basic training parameters
    save_directory="dr-grpo-math-solver",
    num_train_epochs=3,
    total_batch_size=8,
    learning_rate=5e-7,
    max_prompt_length=1024,
    max_completion_length=1024,
    num_return_sequences=4,
    
    # DR GRPO-specific: Constant normalization
    use_constant_normalization=True,
    constant_normalization_factor=1.0,
    
    # DR GRPO-specific: Disable std scaling
    disable_std_scaling=True,
    advantage_epsilon=1e-4,
    
    # Optional: Advantage whitening for stability
    use_advantage_whitening=False,
    whitening_epsilon=1e-8,
    
    # Standard GRPO parameters
    beta=0.04,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
)
```

## Complete Training Example

```python
import easydel as ed
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

def main():
    # 1. Load model and apply LoRA
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=jnp.bfloat16,
        auto_shard_model=True,
    )
    model = model.apply_lora_to_layers(
        rank=16, 
        target_modules=".*q_proj.*"
    )
    
    # 2. Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    
    # 3. Define reward functions
    def format_reward(prompts, completions, **kwargs):
        """Reward proper XML formatting"""
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            has_think_tags = "<think>" in text and "</think>" in text
            rewards.append(0.5 if has_think_tags else 0.0)
        return rewards
    
    def correctness_reward(prompts, completions, batch, **kwargs):
        """Reward correct mathematical answers"""
        rewards = []
        answers = batch.get("answer", [""] * len(completions))
        
        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            if "####" in text:
                predicted = text.split("####")[1].strip()
                correct = answers[i % len(answers)]
                rewards.append(2.0 if predicted == correct else 0.0)
            else:
                rewards.append(0.0)
        return rewards
    
    # 4. Data tokenization function
    def tokenize_function(batch, tokenizer, tools):
        prompts = []
        for question in batch["question"]:
            prompts.append([
                {"role": "system", "content": "Solve step by step. Use <think> tags for reasoning. End with #### [answer]."},
                {"role": "user", "content": question}
            ])
        
        return tokenizer(
            prompts,
            return_tensors="np",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
    
    # 5. DAPO Training
    print("Training with DAPO...")
    dapo_config = ed.DAPOConfig(
        save_directory="dapo-results",
        num_train_epochs=2,
        total_batch_size=4,
        learning_rate=5e-7,
        num_return_sequences=4,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        enable_dynamic_sampling=True,
        use_token_level_loss=True,
        beta=0.0,  # No KL penalty
    )
    
    dapo_trainer = ed.DAPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        processing_class=tokenizer,
        train_dataset=dataset.select(range(100)),  # Small subset for demo
        arguments=dapo_config,
        data_tokenize_fn=tokenize_function,
    )
    
    dapo_trainer.train()
    
    # 6. DR GRPO Training (on fresh model)
    print("Training with DR GRPO...")
    model_dr = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=jnp.bfloat16,
        auto_shard_model=True,
    )
    model_dr = model_dr.apply_lora_to_layers(rank=16, target_modules=".*q_proj.*")
    
    dr_grpo_config = ed.DRGRPOConfig(
        save_directory="dr-grpo-results",
        num_train_epochs=2,
        total_batch_size=4,
        learning_rate=5e-7,
        num_return_sequences=4,
        use_constant_normalization=True,
        disable_std_scaling=True,
        beta=0.04,  # Standard KL penalty
    )
    
    dr_grpo_trainer = ed.DRGRPOTrainer(
        model=model_dr,
        reward_funcs=[format_reward, correctness_reward],
        processing_class=tokenizer,
        train_dataset=dataset.select(range(100)),
        arguments=dr_grpo_config,
        data_tokenize_fn=tokenize_function,
    )
    
    dr_grpo_trainer.train()

if __name__ == "__main__":
    main()
```

## Advanced Configurations

### DAPO for Long Reasoning Chains

```python
# Optimized for complex mathematical reasoning
dapo_long_config = ed.DAPOConfig(
    # Extended sequences for complex reasoning
    max_prompt_length=2048,
    max_completion_length=2048,
    
    # More aggressive exploration
    clip_ratio_high=0.32,
    temperature=0.9,
    
    # Strong dynamic sampling
    enable_dynamic_sampling=True,
    max_resample_attempts=5,
    min_accuracy_variance=0.15,
    
    # Overlong handling for very long solutions
    enable_overlong_reward_shaping=True,
    overlong_buffer_length=8192,
    overlong_penalty_scale=0.05,  # Gentle penalty
    
    # Token-level loss crucial for long sequences
    use_token_level_loss=True,
    
    # No KL penalty for maximum exploration
    beta=0.0,
)
```

### DR GRPO for Efficient Training

```python
# Optimized for compute efficiency and stability
dr_grpo_efficient_config = ed.DRGRPOConfig(
    # Smaller batch size for faster iteration
    total_batch_size=2,
    num_return_sequences=8,  # More candidates per batch
    
    # Constant normalization for stability
    use_constant_normalization=True,
    constant_normalization_factor=0.5,  # Scale down for stability
    
    # No std scaling for equal treatment
    disable_std_scaling=True,
    advantage_epsilon=1e-3,  # Slightly larger for numerical stability
    
    # Advantage whitening for very stable training
    use_advantage_whitening=True,
    whitening_epsilon=1e-6,
    
    # Moderate KL penalty
    beta=0.02,
)
```

## Monitoring and Debugging

### DAPO-Specific Metrics

```python
# Monitor DAPO training progress
def log_dapo_metrics(trainer):
    """Log DAPO-specific training metrics"""
    metrics_to_track = [
        "clipped_fraction",        # How often clipping is applied
        "mean_ratio",             # Policy ratio distribution
        "resample_attempts",      # Dynamic sampling frequency
        "clip_ratio_high",        # Upper clipping bound
        "dynamic_sampling_enabled", # Sampling status
        "token_level_loss",       # Loss computation method
    ]
    
    # Use with Weights & Biases
    dapo_config.use_wandb = True
    dapo_config.wandb_project = "dapo-math-reasoning"
```

### DR GRPO-Specific Metrics

```python
# Monitor DR GRPO bias corrections
def log_dr_grpo_metrics(trainer):
    """Log DR GRPO-specific training metrics"""
    metrics_to_track = [
        "constant_normalization",         # Normalization method
        "constant_normalization_factor",  # Normalization value
        "disable_std_scaling",            # Std scaling status
        "advantage_whitening",            # Whitening application
    ]
    
    dr_grpo_config.use_wandb = True
    dr_grpo_config.wandb_project = "dr-grpo-math-reasoning"
```

## Performance Comparison

Based on research results:

| Method | AIME 2024 Score | Training Efficiency | Best Use Case |
|--------|----------------|-------------------|---------------|
| GRPO | ~35 points | Baseline | General RL training |
| DAPO | **50 points** | 50% fewer steps | Complex reasoning, long chains |
| DR GRPO | **43.3 points** | 27 hours on 8×A100 | Efficient training, quick iteration |

## Troubleshooting

### DAPO Issues

1. **Low diversity in generations**
   - Increase `temperature` and `clip_ratio_high`
   - Enable dynamic sampling with more attempts
   - Check `min_accuracy_variance` threshold

2. **Resampling too frequent**
   - Lower `min_accuracy_variance`
   - Increase reward function diversity
   - Check batch size vs. generation diversity

### DR GRPO Issues

1. **Training instability**
   - Enable advantage whitening
   - Reduce `constant_normalization_factor`
   - Check `advantage_epsilon` value

2. **Slow convergence**
   - Verify `disable_std_scaling=True`
   - Adjust `constant_normalization_factor`
   - Consider higher learning rate

## Hardware Recommendations

### DAPO
- **Minimum**: 16GB VRAM (RTX 4090, A100 40GB)
- **Recommended**: 24GB+ VRAM for dynamic sampling overhead
- **Optimal**: Multi-GPU setup for larger batch sizes

### DR GRPO
- **Minimum**: 8GB VRAM (RTX 3080)
- **Recommended**: 16GB VRAM for comfortable training
- **Optimal**: TPU v3/v4 for maximum efficiency

Both algorithms scale seamlessly from single GPU to TPU clusters with EasyDeL's JAX-based architecture.

## Next Steps

1. **Experiment with hybrid approaches**: Combine DAPO and DR GRPO techniques
2. **Domain adaptation**: Apply to your specific reasoning tasks
3. **Curriculum learning**: Gradually increase problem difficulty
4. **Multi-objective optimization**: Balance multiple reward signals

These advanced GRPO variants represent the current state-of-the-art in LLM reinforcement learning, achieving unprecedented performance on mathematical reasoning benchmarks while maintaining training efficiency. 