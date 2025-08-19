#include "test_common.h"

// Forward declarations from all test modules
// Basic tests
extern int test_basic_backend_init();
extern int test_legacy_flat_config();
extern int test_enhanced_nested_config();
extern int test_legacy_model_config();
extern int test_enhanced_model_config();

// Inference tests
extern int test_basic_inference();
extern int test_advanced_sampling();
extern int test_dynamic_runtime_parameters();

// Session tests
extern int test_session_management();
extern int test_auto_session_cleanup();
extern int test_concurrency_management();

// Logging tests
extern int test_logging_configuration();
extern int test_advanced_logging_features();
extern int test_file_logging();

// Model tests
extern int test_safe_model_switch();

//Add forward declarations for LoRA tests from test_lora.c
extern int test_lora_basic_loading();
extern int test_lora_single_adapter();
extern int test_lora_multi_adapter();
extern int test_lora_dynamic_loading();
extern int test_lora_scaling();
extern int test_lora_error_handling();
extern int test_lora_runtime_override();
extern int test_lora_performance();

// Stopping criteria tests
extern int test_advanced_stopping_criteria();
extern int test_grammar_based_stopping();
extern int test_dynamic_timeout_stopping();
extern int test_token_pattern_stopping();
extern int test_advanced_stopping_integration();

// Error handling tests
extern int test_error_handling();
extern int test_phase42_backend_init();
extern int test_task_queue_interface();
extern int test_phase42_concurrent_access();
extern int test_advanced_task_queue_config();
extern int test_dangerous_edge_cases();

int main() {
    printf("🚀 WASI-NN Backend Comprehensive Test Suite (Modular)\n");
    printf("============================================================\n");
    printf("Testing Enhanced Configuration System with Modular Architecture\n");
    printf("============================================================\n");

    // Initialize library
    if (!setup_library()) {
        printf("❌ FATAL: Failed to setup library\n");
        return EXIT_FAILURE;
    }

    // Run all test sections with modular test files
    TEST_SECTION("Core Functionality Tests (test_basic.c)");
    RUN_TEST("Basic Backend Initialization", test_basic_backend_init);
    RUN_TEST("Legacy Flat Configuration", test_legacy_flat_config);
    RUN_TEST("Enhanced Nested Configuration", test_enhanced_nested_config);
    RUN_TEST("Legacy Model Configuration", test_legacy_model_config);
    RUN_TEST("Enhanced Model Configuration with GPU", test_enhanced_model_config);

    TEST_SECTION("Inference and AI Functionality Tests (test_inference.c)");
    RUN_TEST("Basic Inference Test", test_basic_inference);
    RUN_TEST("Advanced Sampling Parameters", test_advanced_sampling);
    RUN_TEST("Dynamic Runtime Parameters", test_dynamic_runtime_parameters);

    TEST_SECTION("Session Management Tests (test_session.c)");
    RUN_TEST("Session Management and Chat History", test_session_management);
    RUN_TEST("Auto Session Cleanup Validation", test_auto_session_cleanup);
    RUN_TEST("Concurrency Management", test_concurrency_management);

    TEST_SECTION("Advanced Logging System Tests (test_logging.c)");
    RUN_TEST("Basic Logging Configuration", test_logging_configuration);
    RUN_TEST("Advanced Logging Features", test_advanced_logging_features);
    RUN_TEST("File Logging and Structured Output", test_file_logging);

    TEST_SECTION("Model Management Tests (test_model.c)");
    RUN_TEST("Safe Model Switch", test_safe_model_switch);

    TEST_SECTION("Advanced Stopping Criteria Tests (test_stopping.c)");
    RUN_TEST("Advanced Stopping Criteria Configuration", test_advanced_stopping_criteria);
    RUN_TEST("Grammar-Based Stopping Conditions", test_grammar_based_stopping);
    RUN_TEST("Dynamic Timeout and Context-Aware Stopping", test_dynamic_timeout_stopping);
    RUN_TEST("Token-Based and Pattern Stopping Conditions", test_token_pattern_stopping);
    RUN_TEST("Advanced Stopping Criteria Integration", test_advanced_stopping_integration);
    
    TEST_SECTION("LoRA Adapter Tests (test_lora.c)");
    RUN_TEST("Basic LoRA Loading", test_lora_basic_loading);
    RUN_TEST("Single LoRA Adapter", test_lora_single_adapter);
    RUN_TEST("Multiple LoRA Adapters", test_lora_multi_adapter);
    RUN_TEST("Dynamic LoRA Loading", test_lora_dynamic_loading);
    RUN_TEST("LoRA Scaling", test_lora_scaling);
    RUN_TEST("LoRA Error Handling", test_lora_error_handling);
    RUN_TEST("LoRA Runtime Override", test_lora_runtime_override);
    RUN_TEST("LoRA Performance Impact", test_lora_performance);

    TEST_SECTION("Error Handling and Task Management Tests (test_error.c)");
    RUN_TEST("Error Handling and Edge Cases", test_error_handling);
    RUN_TEST("Phase 4.2 Backend Initialization with Task Queue", test_phase42_backend_init);
    RUN_TEST("Task Queue Interface Testing", test_task_queue_interface);
    RUN_TEST("Phase 4.2 Concurrent Thread Access", test_phase42_concurrent_access);
    RUN_TEST("Advanced Task Queue Configuration", test_advanced_task_queue_config);
    RUN_TEST("Dangerous Edge Cases (with Signal Protection)", test_dangerous_edge_cases);

    // Final report
    printf("\n======================================================================\n");
    printf("🏁 MODULAR TEST SUITE SUMMARY\n");
    printf("======================================================================\n");
    printf("Total Tests: %d\n", test_count);
    printf("✅ Passed:   %d\n", test_passed);
    printf("❌ Failed:   %d\n", test_failed);
    
    printf("\n📁 Test File Organization:\n");
    printf("• test_common.h/c     - Common framework and utilities\n");
    printf("• test_basic.c        - Backend initialization and configuration\n");
    printf("• test_inference.c    - Inference execution and sampling parameters\n");
    printf("• test_session.c      - Session management and concurrency control\n");
    printf("• test_logging.c      - Logging system configuration and output\n");
    printf("• test_model.c        - Model switching functionality\n");
    printf("• test_stopping.c     - Advanced stopping criteria\n");
    printf("• test_lora.c         - LoRA adapter loading and inference\n");
    printf("• test_error.c        - Error handling, task queues, and edge cases\n");
    printf("• main.c              - Test runner and coordinator\n");
    
    if (test_failed == 0) {
        printf("\n🎉 ALL TESTS PASSED! 🎉\n");
        printf("✅ Modular test architecture working perfectly!\n");
        printf("✅ Enhanced Configuration System fully functional!\n");
        printf("✅ Advanced Concurrency and Task Management operational!\n");
        printf("✅ Advanced Memory Management working automatically!\n");
        printf("✅ Advanced Logging System with file output working!\n");
        printf("✅ Stable Model Switching without crashes or leaks!\n");
        printf("✅ Advanced Stopping Criteria with all trigger types!\n");
        printf("✅ Comprehensive Error Handling and Edge Case Protection!\n");
        printf("✅ GPU acceleration enabled and optimized!\n");
        printf("✅ Full backward compatibility maintained!\n");
        printf("✅ Thread-safe concurrent access validated!\n");
        printf("✅ Auto-cleanup session management working!\n");
        printf("✅ Task queue system with priority scheduling!\n");
        printf("✅ Signal protection for dangerous edge cases!\n");
        printf("✅ LoRA adapter support fully integrated!\n");
        printf("✅ Single and multi-adapter loading validated!\n");
        printf("✅ Dynamic LoRA scaling and runtime overrides working!\n");
    } else {
        printf("\n⚠️  Some tests failed. Please review the output above.\n");
        printf("Check individual test files for specific failures.\n");
    }
    
    printf("======================================================================\n");

    // Cleanup with safety checks
    if (handle) {
        // Give some time for any background GPU operations to complete
        usleep(100000);  // 100ms delay
        
        // Safely close the dynamic library
        int dlclose_result = dlclose(handle);
        if (dlclose_result != 0) {
            printf("⚠️  Warning: dlclose returned error: %s\n", dlerror());
        }
        handle = NULL;
    }

    // Force a small delay before program exit to allow GPU cleanup
    usleep(50000);  // 50ms delay

    return (test_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
