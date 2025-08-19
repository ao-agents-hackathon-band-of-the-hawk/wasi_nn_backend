#include "test_common.h"

// Global variables
jmp_buf segfault_jmp;
volatile sig_atomic_t segfault_occurred = 0;

int test_count = 0;
int test_passed = 0;
int test_failed = 0;

void *handle = NULL;
init_backend_func_t wasi_init_backend = NULL;
init_backend_with_config_func_t wasi_init_backend_with_config = NULL;
load_by_name_with_configuration_func_t wasi_load_by_name_with_config = NULL;
init_execution_context_func_t wasi_init_execution_context = NULL;
init_execution_context_with_session_id_func_t wasi_init_execution_context_with_session_id = NULL;
close_execution_context_func_t wasi_close_execution_context = NULL;
run_inference_func_t wasi_run_inference = NULL;
set_input_func_t wasi_set_input = NULL;
compute_func_t wasi_compute = NULL;
get_output_func_t wasi_get_output = NULL;
deinit_backend_func_t wasi_deinit_backend = NULL;

const char *MODEL_FILE = "./models/qwen2.5-14b-instruct-q2_k.gguf";
const char *MODEL_CONFIG = "{\"n_gpu_layers\":0,\"ctx_size\":512,\"n_predict\":10}";
tensor_dimensions global_text_dims = {NULL, 0};

void segfault_handler(int sig) {
    segfault_occurred = 1;
    longjmp(segfault_jmp, 1);
}

void setup_tensor(tensor *t, const char *data) {
    t->data = (uint8_t *)data;
    t->dimensions = &global_text_dims;
    t->type = u8;
}

int setup_library(void) {
    // Load the shared library
    handle = dlopen("../build/libwasi_nn_backend.so", RTLD_LAZY);
    if (handle == NULL) {
        printf("Error loading library: %s\n", dlerror());
    }
    ASSERT(handle != NULL, "Failed to load shared library");

    // Get function pointers
    *(void **)(&wasi_init_backend) = dlsym(handle, "init_backend");
    *(void **)(&wasi_init_backend_with_config) = dlsym(handle, "init_backend_with_config");
    *(void **)(&wasi_load_by_name_with_config) = dlsym(handle, "load_by_name_with_config");
    *(void **)(&wasi_init_execution_context) = dlsym(handle, "init_execution_context");
    *(void **)(&wasi_init_execution_context_with_session_id) = dlsym(handle, "init_execution_context_with_session_id");
    *(void **)(&wasi_close_execution_context) = dlsym(handle, "close_execution_context");
    *(void **)(&wasi_run_inference) = dlsym(handle, "run_inference");
    *(void **)(&wasi_set_input) = dlsym(handle, "set_input");
    *(void **)(&wasi_compute) = dlsym(handle, "compute");
    *(void **)(&wasi_get_output) = dlsym(handle, "get_output");
    *(void **)(&wasi_deinit_backend) = dlsym(handle, "deinit_backend");

    char *error = dlerror();
    ASSERT(error == NULL, "Failed to load function symbols");

    printf("✅ Library loaded successfully\n");
    return 1;
}
