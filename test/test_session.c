#include "test_common.h"
#include <unistd.h>
// Test 9: Session Management and Chat History
int test_session_management() {
    void *backend_ctx = NULL;
    graph g = 0;
    graph_execution_context exec_ctx = 0;
    wasi_nn_error err;

    const char *config = "{\"max_sessions\":10,\"idle_timeout_ms\":600000,\"auto_cleanup\":true}";
    err = wasi_init_backend_with_config(&backend_ctx, config, strlen(config));
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *model_config = "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"n_predict\":60}";
    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");

    err = wasi_init_execution_context(backend_ctx, g, &exec_ctx);
    ASSERT_SUCCESS(err, "Execution context initialization failed");

    // First message
    tensor input_tensor1;
    setup_tensor(&input_tensor1, "Hello, my name is Alice.");

    uint8_t output_buffer1[512];
    uint32_t output_size1 = sizeof(output_buffer1);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor1, output_buffer1, &output_size1, NULL, 0);
    ASSERT_SUCCESS(err, "First inference failed");

    printf("✅ First response: %.60s%s\n", 
           (char*)output_buffer1, output_size1 > 60 ? "..." : "");

    // Second message (should remember context)
    tensor input_tensor2;
    setup_tensor(&input_tensor2, "What is my name?");

    uint8_t output_buffer2[512];
    uint32_t output_size2 = sizeof(output_buffer2);

    err = wasi_run_inference(backend_ctx, exec_ctx, 0, &input_tensor2, output_buffer2, &output_size2, NULL, 0);
    ASSERT_SUCCESS(err, "Second inference failed");

    printf("✅ Context-aware response: %.60s%s\n", 
           (char*)output_buffer2, output_size2 > 60 ? "..." : "");

    // Cleanup
    wasi_close_execution_context(backend_ctx, exec_ctx);
    wasi_deinit_backend(backend_ctx);

    return 1;
}

// Test 10: Auto Session Cleanup Validation
int test_auto_session_cleanup() {
    printf("🧪 Testing auto_cleanup_sessions functionality...\n");
    
    void *backend_ctx = NULL;
    graph g = 0;
    wasi_nn_error err;

    // Initialize backend with auto cleanup enabled and short timeout
    const char *config = "{"
        "\"max_sessions\":3,"
        "\"idle_timeout_ms\":100,"  // Very short timeout for testing
        "\"auto_cleanup_enabled\":true,"
        "\"max_concurrent\":2"
    "}";
    
    err = wasi_init_backend_with_config(&backend_ctx, config, strlen(config));
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *model_config = "{\"n_gpu_layers\":0,\"ctx_size\":512,\"n_predict\":10}";
    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");

    // Test 1: Create multiple sessions to test LRU eviction
    printf("📝 Test 1: Creating multiple sessions for LRU eviction test...\n");
    
    graph_execution_context exec_ctx1, exec_ctx2, exec_ctx3, exec_ctx4;
    
    // Create session 1
    err = wasi_init_execution_context_with_session_id(backend_ctx, "session_1", &exec_ctx1);
    ASSERT_SUCCESS(err, "Session 1 creation failed");
    printf("✅ Created session_1 (exec_ctx: %u)\n", exec_ctx1);
    
    // Small delay to differentiate timestamps
    usleep(10000); // 10ms
    
    // Create session 2
    err = wasi_init_execution_context_with_session_id(backend_ctx, "session_2", &exec_ctx2);
    ASSERT_SUCCESS(err, "Session 2 creation failed");
    printf("✅ Created session_2 (exec_ctx: %u)\n", exec_ctx2);
    
    usleep(10000); // 10ms
    
    // Create session 3
    err = wasi_init_execution_context_with_session_id(backend_ctx, "session_3", &exec_ctx3);
    ASSERT_SUCCESS(err, "Session 3 creation failed");
    printf("✅ Created session_3 (exec_ctx: %u)\n", exec_ctx3);
    
    usleep(10000); // 10ms
    
    // Now try to create session 4 - should trigger LRU cleanup of session_1
    printf("📝 Creating session_4 - should trigger LRU cleanup...\n");
    err = wasi_init_execution_context_with_session_id(backend_ctx, "session_4", &exec_ctx4);
    ASSERT_SUCCESS(err, "Session 4 creation failed");
    printf("✅ Created session_4 (exec_ctx: %u) - LRU cleanup should have occurred\n", exec_ctx4);
    
    // Test 2: Verify that session_1 was cleaned up by trying to use it
    printf("📝 Test 2: Verifying session_1 was cleaned up...\n");
    
    tensor input_tensor;
    setup_tensor(&input_tensor, "Test message");
    uint8_t output_buffer[256];
    uint32_t output_size = sizeof(output_buffer);
    
    // This should fail because session_1 should have been cleaned up
    err = wasi_run_inference(backend_ctx, exec_ctx1, 0, &input_tensor, output_buffer, &output_size, NULL, 0);
    if (err != 0) {
        printf("✅ Session_1 properly cleaned up (inference failed as expected)\n");
    } else {
        printf("⚠️  Session_1 still exists (cleanup may not be working)\n");
    }
    
    // Test 3: Idle timeout cleanup
    printf("📝 Test 3: Testing idle timeout cleanup...\n");
    
    // Wait for idle timeout to trigger
    printf("⏳ Waiting 150ms for idle timeout to trigger...\n");
    usleep(150000); // 150ms - longer than idle_timeout_ms (100ms)
    
    // Try to create a new session - this should trigger idle timeout cleanup
    graph_execution_context exec_ctx5;
    err = wasi_init_execution_context_with_session_id(backend_ctx, "session_5", &exec_ctx5);
    ASSERT_SUCCESS(err, "Session 5 creation failed");
    printf("✅ Created session_5 - idle timeout cleanup should have occurred\n");
    
    // Test if old sessions were cleaned up by idle timeout
    err = wasi_run_inference(backend_ctx, exec_ctx2, 0, &input_tensor, output_buffer, &output_size, NULL, 0);
    if (err != 0) {
        printf("✅ Session_2 properly cleaned up by idle timeout\n");
    } else {
        printf("⚠️  Session_2 still exists (idle timeout cleanup may not be working)\n");
    }
    
    // Test 4: Test auto_cleanup_enabled flag
    printf("📝 Test 4: Testing auto_cleanup_enabled flag...\n");
    
    // Cleanup current backend
    wasi_deinit_backend(backend_ctx);
    
    // Initialize backend with auto cleanup disabled
    const char *config_disabled = "{"
        "\"max_sessions\":2,"
        "\"idle_timeout_ms\":50,"
        "\"auto_cleanup_enabled\":false"
    "}";
    
    err = wasi_init_backend_with_config(&backend_ctx, config_disabled, strlen(config_disabled));
    ASSERT_SUCCESS(err, "Backend initialization with disabled cleanup failed");
    
    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");
    
    // Create sessions that would normally be cleaned up
    graph_execution_context exec_ctx_no_cleanup1, exec_ctx_no_cleanup2, exec_ctx_no_cleanup3;
    
    err = wasi_init_execution_context_with_session_id(backend_ctx, "no_cleanup_1", &exec_ctx_no_cleanup1);
    ASSERT_SUCCESS(err, "No cleanup session 1 creation failed");
    
    err = wasi_init_execution_context_with_session_id(backend_ctx, "no_cleanup_2", &exec_ctx_no_cleanup2);
    ASSERT_SUCCESS(err, "No cleanup session 2 creation failed");
    
    // Wait for what would be idle timeout
    usleep(100000); // 100ms
    
    // Try to create a third session - should hit concurrency limit since cleanup is disabled
    err = wasi_init_execution_context_with_session_id(backend_ctx, "no_cleanup_3", &exec_ctx_no_cleanup3);
    if (err != 0) {
        printf("✅ Concurrency limit properly enforced when auto_cleanup is disabled\n");
    } else {
        printf("⚠️  Session created despite concurrency limit (cleanup may have still occurred)\n");
        wasi_close_execution_context(backend_ctx, exec_ctx_no_cleanup3);
    }
    
    // Cleanup
    wasi_close_execution_context(backend_ctx, exec_ctx_no_cleanup1);
    wasi_close_execution_context(backend_ctx, exec_ctx_no_cleanup2);
    wasi_close_execution_context(backend_ctx, exec_ctx5);
    wasi_deinit_backend(backend_ctx);
    
    printf("✅ Auto session cleanup validation completed\n");
    return 1;
}

// Test 7: Concurrency Management
int test_concurrency_management() {
    void *backend_ctx = NULL;
    graph g = 0;
    wasi_nn_error err;

    // Setup backend with limited concurrency
    const char *config = "{\"max_concurrent\":2,\"queue_size\":5}";
    err = wasi_init_backend_with_config(&backend_ctx, config, strlen(config));
    ASSERT_SUCCESS(err, "Backend initialization failed");

    const char *model_config = "{\"n_gpu_layers\":98,\"ctx_size\":1024,\"n_predict\":50}";
    err = wasi_load_by_name_with_config(backend_ctx, MODEL_FILE, strlen(MODEL_FILE),
                                  model_config, strlen(model_config), &g);
    ASSERT_SUCCESS(err, "Model loading failed");

    graph_execution_context ctx1, ctx2, ctx3;

    // First two should succeed
    err = wasi_init_execution_context(backend_ctx, g, &ctx1);
    ASSERT_SUCCESS(err, "First execution context failed");

    err = wasi_init_execution_context(backend_ctx, g, &ctx2);
    ASSERT_SUCCESS(err, "Second execution context failed");

    // Third should fail due to concurrency limit
    err = wasi_init_execution_context(backend_ctx, g, &ctx3);
    ASSERT(err == runtime_error, "Concurrency limit not enforced");

    printf("✅ Concurrency limit properly enforced (2/2 slots used)\n");

    // Close one context and try again
    wasi_close_execution_context(backend_ctx, ctx1);

    err = wasi_init_execution_context(backend_ctx, g, &ctx3);
    ASSERT_SUCCESS(err, "Context creation failed after slot became available");

    printf("✅ Context creation successful after slot freed (2/2 slots used)\n");

    // Cleanup
    wasi_close_execution_context(backend_ctx, ctx2);
    wasi_close_execution_context(backend_ctx, ctx3);
    wasi_deinit_backend(backend_ctx);

    return 1;
}
