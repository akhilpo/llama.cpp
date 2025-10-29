# MoE Expert Logging Example

This example demonstrates how to use the MoE (Mixture of Experts) expert logging feature in llama.cpp to track which experts are selected for each token during inference. This is particularly useful for analyzing GPT-OSS and other MoE architecture models.

## Features

The MoE expert logging system provides:

1. **Per-token expert selection tracking**: Logs which experts are selected for each token at each layer
2. **Expert usage statistics**: Shows how many times each expert was used
3. **Consecutive expert usage analysis**: Tracks patterns of experts being used consecutively across tokens
4. **Detailed logging**: Provides a complete log of expert selections for analysis

## API Functions

### `llama_moe_expert_logging_enable(ctx, enable)`
Enable or disable MoE expert logging. When enabled, the system will track all expert selections during inference.

```cpp
llama_moe_expert_logging_enable(ctx, true);  // Enable logging
llama_moe_expert_logging_enable(ctx, false); // Disable logging
```

### `llama_moe_expert_logging_is_enabled(ctx)`
Check if MoE expert logging is currently enabled.

```cpp
if (llama_moe_expert_logging_is_enabled(ctx)) {
    printf("Expert logging is enabled\n");
}
```

### `llama_moe_expert_logging_clear(ctx)`
Clear all logged expert selection data.

```cpp
llama_moe_expert_logging_clear(ctx);
```

### `llama_moe_expert_logging_print_stats(ctx)`
Print comprehensive statistics about expert selections including:
- Total tokens and layers processed
- Expert usage counts (how many times each expert was used)
- Consecutive expert usage patterns
- Detailed per-token expert selections

```cpp
llama_moe_expert_logging_print_stats(ctx);
```

## Usage

### Building the Example

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --target moe-expert-logging
```

### Running the Example

```bash
./build/bin/moe-expert-logging <model_path> [prompt]
```

Example:
```bash
./build/bin/moe-expert-logging ./models/gpt-oss-120b.gguf "Explain mixture of experts"
```

## Example Output

```
Enabling MoE expert logging...
Prompt: Explain mixture of experts
Number of tokens: 5

Processing tokens...

Inference complete!

=== MoE Expert Selection Statistics ===
Total tokens processed: 5
Total layers with MoE: 24

Expert Usage Count:
  Expert 0: 12 times
  Expert 1: 8 times
  Expert 2: 15 times
  Expert 3: 10 times
  Expert 4: 6 times
  Expert 5: 14 times
  Expert 6: 9 times
  Expert 7: 11 times

Consecutive Expert Usage (across consecutive tokens):
  1 expert(s) used consecutively: 45 occurrences
  2 expert(s) used consecutively: 28 occurrences
  Total consecutive patterns: 73

Detailed Expert Selections:
  Layer  0, Token   0: Experts [2, 5]
  Layer  0, Token   1: Experts [0, 7]
  Layer  0, Token   2: Experts [3, 5]
  Layer  0, Token   3: Experts [1, 2]
  Layer  0, Token   4: Experts [6, 7]
  Layer  1, Token   0: Experts [1, 4]
  ...
=======================================
```

## Understanding the Statistics

### Expert Usage Count
Shows how many times each expert was selected across all tokens and layers. This helps identify:
- Most frequently used experts
- Underutilized experts
- Expert specialization patterns

### Consecutive Expert Usage
Tracks how many experts appear in both consecutive tokens at the same layer. This reveals:
- Temporal consistency in expert selection
- Context-dependent expert switching patterns
- Expert stability across token sequences

## Implementation Notes

- Currently optimized for CPU backend
- Works with all MoE architectures supported by llama.cpp (GPT-OSS, DeepSeek V3, Qwen MoE, etc.)
- Minimal performance overhead when disabled
- Expert selections are logged after each batch processing

## Use Cases

1. **Model Analysis**: Understand which experts are being utilized for different types of input
2. **Performance Optimization**: Identify frequently used experts for targeted optimization
3. **Research**: Study expert specialization and selection patterns
4. **Debugging**: Verify correct MoE routing behavior
5. **Model Pruning**: Identify underutilized experts that might be candidates for pruning

## Integrating into Your Code

```cpp
#include "llama.h"

int main() {
    // Initialize model and context
    llama_model * model = llama_model_load_from_file("model.gguf", params);
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    // Enable expert logging
    llama_moe_expert_logging_enable(ctx, true);

    // Run inference
    llama_decode(ctx, batch);

    // Print statistics
    llama_moe_expert_logging_print_stats(ctx);

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
```
