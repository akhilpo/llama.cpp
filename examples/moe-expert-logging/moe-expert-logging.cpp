// Example of using MoE expert logging for GPT-OSS and other MoE architectures
// This demonstrates how to track which experts are selected for each token during inference

#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [prompt]\n", argv[0]);
        fprintf(stderr, "Example: %s ./models/gpt-oss-model.gguf \"Hello world\"\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt = argc > 2 ? argv[2] : "Once upon a time";

    // Initialize llama backend
    llama_backend_init();

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path);
        return 1;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = 4;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Enable MoE expert logging
    printf("Enabling MoE expert logging...\n");
    llama_moe_expert_logging_enable(ctx, true);

    // Tokenize the prompt
    std::vector<llama_token> tokens;
    const int n_prompt_tokens = -llama_tokenize(
        model,
        prompt,
        strlen(prompt),
        nullptr,
        0,
        true,  // add_special
        true   // parse_special
    );

    tokens.resize(n_prompt_tokens);
    llama_tokenize(
        model,
        prompt,
        strlen(prompt),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    printf("Prompt: %s\n", prompt);
    printf("Number of tokens: %d\n", (int)tokens.size());

    // Create a batch
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    // Add tokens to batch
    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    // Process the batch
    printf("\nProcessing tokens...\n");
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Failed to decode\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    printf("\nInference complete!\n\n");

    // Print expert selection statistics
    llama_moe_expert_logging_print_stats(ctx);

    // Generate a few tokens to see more expert selections
    printf("\n=== Generating additional tokens ===\n\n");

    for (int i = 0; i < 10; i++) {
        // Get logits for the last token
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        // Simple greedy sampling - pick the token with highest probability
        int vocab_size = llama_n_vocab(model);
        llama_token next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < vocab_size; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }

        // Print the generated token
        char token_str[128];
        llama_token_to_piece(model, next_token, token_str, sizeof(token_str), 0, true);
        printf("%s", token_str);
        fflush(stdout);

        // Add the new token to the batch
        llama_batch_clear(batch);
        llama_batch_add(batch, next_token, tokens.size() + i, {0}, true);

        // Decode the next token
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nFailed to decode next token\n");
            break;
        }
    }

    printf("\n\n=== Final Expert Selection Statistics ===\n");
    llama_moe_expert_logging_print_stats(ctx);

    // Cleanup
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
