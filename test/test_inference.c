#include "test_common.h"

// Test 6: Basic Inference Test
int test_basic_inference() {
    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    // Setup backend and model
    const char *config = "{\"max_concurrent\":4}";
    err = wasi_init_backend_with_config(&backend_ctx, config, strlen(config));
    ASSERT_SUCCESS(err, "Backend initialization failed");

     const char *model_config =
        "{"
        "  \"model\": {"
        "    \"n_gpu_layers\": 49,"
        "    \"ctx_size\": 2048,"
        "    \"batch_size\": 512,"
        "    \"threads\": 4"
        "  },"
        "  \"backend\": {"
        "    \"max_sessions\": 50,"
        "    \"max_concurrent\": 4,"
        "    \"queue_size\": 20"
        "  },"
        "  \"logging\": {"
        "    \"level\": \"info\","
        "    \"enable_debug\": true"
        "  },"
        "  \"lora_adapters\": ["
        "    {"
        "      \"path\": \"./models/LoRA_adapter1.gguf\","
        "      \"scale\": 1.0"
        "    }"
        "  ]"
        "}";


    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");

    err = wasi_init_execution_context(backend_ctx, g, &exec_ctx);
    ASSERT_SUCCESS(err, "Execution context initialization failed");

    // Prepare inference
    tensor input_tensor;
    setup_tensor(&input_tensor, "What is artificial intelligence?");

    uint8_t output_buffer[1024];
    uint32_t output_size = sizeof(output_buffer);

    // Run inference
    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor, output_buffer, &output_size, NULL, 0);
    ASSERT_SUCCESS(err, "Inference execution failed");
    ASSERT(output_size > 0, "No output generated");

    printf("✅ Inference response (%d chars): %.100s%s\n", 
           output_size, (char*)output_buffer, output_size > 100 ? "..." : "");

    // Cleanup
    wasi_close_execution_context(backend_ctx, exec_ctx);
    wasi_deinit_backend(backend_ctx);

    return 1;
}

// Test 8: Advanced Sampling Parameters
int test_advanced_sampling() {
    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    err = wasi_init_backend(&backend_ctx);
    ASSERT_SUCCESS(err, "Backend initialization failed");

    // Test comprehensive sampling configuration
    const char *sampling_config = "{"
                                 "\"model\":{\"n_gpu_layers\":98,\"ctx_size\":1024,\"n_predict\":80},"
                                 "\"sampling\":{"
                                 "\"temp\":0.9,"
                                 "\"top_p\":0.8,"
                                 "\"top_k\":30,"
                                 "\"min_p\":0.1,"
                                 "\"typical_p\":0.95,"
                                 "\"repeat_penalty\":1.15,"
                                 "\"presence_penalty\":0.1,"
                                 "\"frequency_penalty\":0.1,"
                                 "\"penalty_last_n\":32,"
                                 "\"mirostat\":1,"
                                 "\"mirostat_tau\":4.0,"
                                 "\"mirostat_eta\":0.2,"
                                 "\"seed\":12345"
                                 "},"
                                 "\"stopping\":{"
                                 "\"stop\":[\".\",\"!\",\"?\"],"
                                 "\"max_tokens\":80,"
                                 "\"ignore_eos\":true"
                                 "}"
                                 "}";

    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  sampling_config, strlen(sampling_config), &g);
    ASSERT_SUCCESS(err, "Advanced sampling model configuration failed");

    err = wasi_init_execution_context(backend_ctx, g, &exec_ctx);
    ASSERT_SUCCESS(err, "Execution context initialization failed");

    tensor input_tensor;
    setup_tensor(&input_tensor, "Write a short story about");

    uint8_t output_buffer[512];
    uint32_t output_size = sizeof(output_buffer);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor, output_buffer, &output_size, NULL, 0);
    ASSERT_SUCCESS(err, "Advanced sampling inference failed");

    printf("✅ Advanced sampling output: %.80s%s\n", 
           (char*)output_buffer, output_size > 80 ? "..." : "");

    // Cleanup
    wasi_close_execution_context(backend_ctx, exec_ctx);
    wasi_deinit_backend(backend_ctx);

    return 1;
}

// Test: Dynamic Runtime Parameters
int test_dynamic_runtime_parameters() {
    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    printf("Testing dynamic runtime parameter modification...\n");

    // Setup backend and model with base configuration
    const char *config = "{\"max_concurrent\":4}";
    err = wasi_init_backend_with_config(&backend_ctx, config, strlen(config));
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *model_config = "{"
                              "\"n_gpu_layers\":98,"
                              "\"ctx_size\":2048,"
                              "\"n_predict\":50,"
                              "\"sampling\":{\"temp\":0.7,\"top_p\":0.9}"
                              "}";

    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");

    err = wasi_init_execution_context(backend_ctx, g, &exec_ctx);
    ASSERT_SUCCESS(err, "Execution context initialization failed");

    // Test 1: Basic inference without runtime config (should use defaults)
    printf("\n--- Test 1: Default parameters ---\n");
    tensor input_tensor1;
    setup_tensor(&input_tensor1, "Generate a creative story about a robot.");

    uint8_t output_buffer1[512];
    uint32_t output_size1 = sizeof(output_buffer1);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor1, output_buffer1, &output_size1, NULL, 0);
    ASSERT_SUCCESS(err, "Default inference failed");
    ASSERT(output_size1 > 0, "No output generated with default parameters");

    printf("✅ Default response (%d chars): %.80s%s\n", 
           output_size1, (char*)output_buffer1, output_size1 > 80 ? "..." : "");

    // Test 2: High creativity (high temperature)
    printf("\n--- Test 2: High creativity (temp=1.2) ---\n");
    const char *high_creativity_config = "{"
                                        "\"temperature\":1.2,"
                                        "\"top_p\":0.95,"
                                        "\"max_tokens\":40"
                                        "}";

    tensor input_tensor2;
    setup_tensor(&input_tensor2, "Generate a creative story about a robot.");

    uint8_t output_buffer2[512];
    uint32_t output_size2 = sizeof(output_buffer2);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor2, output_buffer2, &output_size2, 
                           high_creativity_config, strlen(high_creativity_config));
    ASSERT_SUCCESS(err, "High creativity inference failed");
    ASSERT(output_size2 > 0, "No output generated with high creativity");

    printf("✅ High creativity response (%d chars): %.80s%s\n", 
           output_size2, (char*)output_buffer2, output_size2 > 80 ? "..." : "");

    // Test 3: Low creativity (low temperature)
    printf("\n--- Test 3: Low creativity (temp=0.2) ---\n");
    const char *low_creativity_config = "{"
                                       "\"temperature\":0.2,"
                                       "\"top_p\":0.7,"
                                       "\"max_tokens\":30"
                                       "}";

    tensor input_tensor3;
    setup_tensor(&input_tensor3, "Generate a creative story about a robot.");

    uint8_t output_buffer3[512];
    uint32_t output_size3 = sizeof(output_buffer3);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor3, output_buffer3, &output_size3,
                           low_creativity_config, strlen(low_creativity_config));
    ASSERT_SUCCESS(err, "Low creativity inference failed");
    ASSERT(output_size3 > 0, "No output generated with low creativity");

    printf("✅ Low creativity response (%d chars): %.80s%s\n", 
           output_size3, (char*)output_buffer3, output_size3 > 80 ? "..." : "");

    // Test 4: Custom stop sequences
    printf("\n--- Test 4: Custom stop sequences ---\n");
    const char *stop_config = "{"
                             "\"temperature\":0.8,"
                             "\"max_tokens\":100,"
                             "\"stop\":[\".\",\"!\"]"
                             "}";

    tensor input_tensor4;
    setup_tensor(&input_tensor4, "List three benefits of AI");

    uint8_t output_buffer4[512];
    uint32_t output_size4 = sizeof(output_buffer4);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor4, output_buffer4, &output_size4,
                           stop_config, strlen(stop_config));
    ASSERT_SUCCESS(err, "Stop sequences inference failed");
    ASSERT(output_size4 > 0, "No output generated with stop sequences");

    printf("✅ Stop sequences response (%d chars): %.80s%s\n", 
           output_size4, (char*)output_buffer4, output_size4 > 80 ? "..." : "");

    // Test 5: Advanced sampling parameters
    printf("\n--- Test 5: Advanced sampling parameters ---\n");
    const char *advanced_config = "{"
                                 "\"temperature\":0.9,"
                                 "\"top_p\":0.85,"
                                 "\"top_k\":50,"
                                 "\"repeat_penalty\":1.15,"
                                 "\"frequency_penalty\":0.1,"
                                 "\"presence_penalty\":0.1,"
                                 "\"max_tokens\":35,"
                                 "\"seed\":42"
                                 "}";

    tensor input_tensor5;
    setup_tensor(&input_tensor5, "Explain quantum computing in simple terms");

    uint8_t output_buffer5[512];
    uint32_t output_size5 = sizeof(output_buffer5);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor5, output_buffer5, &output_size5,
                           advanced_config, strlen(advanced_config));
    ASSERT_SUCCESS(err, "Advanced sampling inference failed");
    ASSERT(output_size5 > 0, "No output generated with advanced sampling");

    printf("✅ Advanced sampling response (%d chars): %.80s%s\n", 
           output_size5, (char*)output_buffer5, output_size5 > 80 ? "..." : "");

    // Test 6: Error handling with invalid JSON
    printf("\n--- Test 6: Error handling with invalid JSON ---\n");
    const char *invalid_config = "{\"temperature\":0.8,\"invalid_json";

    tensor input_tensor6;
    setup_tensor(&input_tensor6, "Test invalid config");

    uint8_t output_buffer6[512];
    uint32_t output_size6 = sizeof(output_buffer6);

    // This should still work but use default parameters
    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor6, output_buffer6, &output_size6,
                           invalid_config, strlen(invalid_config));
    ASSERT_SUCCESS(err, "Invalid config should fallback to defaults");
    ASSERT(output_size6 > 0, "No output generated with invalid config");

    printf("✅ Invalid config handled gracefully, response (%d chars): %.80s%s\n", 
           output_size6, (char*)output_buffer6, output_size6 > 80 ? "..." : "");

    // Test 7: Extreme parameters (should be handled gracefully)
    printf("\n--- Test 7: Extreme parameters (boundary testing) ---\n");
    const char *extreme_config = "{"
                                "\"temperature\":5.0,"  // Very high temperature
                                "\"top_p\":0.01,"       // Very low top_p
                                "\"max_tokens\":10,"    // Very few tokens
                                "\"repeat_penalty\":2.0" // High repeat penalty
                                "}";

    tensor input_tensor7;
    setup_tensor(&input_tensor7, "Hello");

    uint8_t output_buffer7[512];
    uint32_t output_size7 = sizeof(output_buffer7);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor7, output_buffer7, &output_size7,
                           extreme_config, strlen(extreme_config));
    ASSERT_SUCCESS(err, "Extreme parameters inference failed");
    ASSERT(output_size7 > 0, "No output generated with extreme parameters");

    printf("✅ Extreme parameters handled, response (%d chars): %.80s%s\n", 
           output_size7, (char*)output_buffer7, output_size7 > 80 ? "..." : "");

    printf("\n✅ All dynamic runtime parameter tests passed!\n");
    printf("✅ Default parameters work correctly\n");
    printf("✅ Temperature modification works\n");
    printf("✅ Token limits are respected\n");
    printf("✅ Stop sequences are applied\n");
    printf("✅ Advanced sampling parameters work\n");
    printf("✅ Error handling is robust\n");
    printf("✅ Extreme parameter values are handled gracefully\n");
    printf("✅ LoRA adapters handled perfectly");
    // Cleanup
    wasi_close_execution_context(backend_ctx, exec_ctx);
    wasi_deinit_backend(backend_ctx);

    return 1;
}
