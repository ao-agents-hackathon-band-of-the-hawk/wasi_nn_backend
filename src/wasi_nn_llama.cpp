/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "../include/wasi_nn_llama.h"
#include "cJSON.h"
#include "utils/logger.h"

// Include llama.cpp headers
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

// Include all server types and structures directly from server.cpp
// We need the complete definitions, not just forward declarations
#include "server/server.cpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>
#include <thread>
#include <condition_variable>

// Enhanced logging macros that work with both old and new systems
#define WASI_NN_LOG_DEBUG(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_DBG("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_DBG_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_INFO(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_INF("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_INFO_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_WARN(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_WRN("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_WARN_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

#define WASI_NN_LOG_ERROR(ctx, fmt, ...) \
  do { \
    if (ctx && ctx->log_initialized) { \
      LOG_ERR("[WASI-NN] " fmt, ##__VA_ARGS__); \
    } else { \
      NN_ERR_PRINTF(fmt, ##__VA_ARGS__); \
    } \
  } while(0)

// Task priority levels for WASI-NN backend
enum wasi_nn_task_priority
{
  WASI_NN_PRIORITY_LOW = 0,
  WASI_NN_PRIORITY_NORMAL = 1,
  WASI_NN_PRIORITY_HIGH = 2,
  WASI_NN_PRIORITY_URGENT = 3
};

// Runtime parameters structure for dynamic inference configuration
struct wasi_nn_runtime_params
{
  // Sampling parameters (most commonly modified at runtime)
  float temperature = -1.0f;           // -1 means use default/existing value
  float top_p = -1.0f;
  int32_t top_k = -1;
  float min_p = -1.0f;
  float typical_p = -1.0f;

  // Penalty parameters
  float repeat_penalty = -1.0f;
  float frequency_penalty = -1.0f;
  float presence_penalty = -1.0f;
  int32_t penalty_last_n = -1;

  // Generation control
  int32_t max_tokens = -1;
  int32_t seed = 1234; // Default seed, can be overridden
  bool ignore_eos = false;  // Default to false, but can be overridden
  bool ignore_eos_set = false;  // Flag to indicate if ignore_eos was explicitly set

  // DRY sampling parameters
  float dry_multiplier = -1.0f;
  float dry_base = -1.0f;
  int32_t dry_allowed_length = -1;
  int32_t dry_penalty_last_n = -1;

  // Dynamic temperature parameters
  float dynatemp_range = -1.0f;
  float dynatemp_exponent = -1.0f;

  // Mirostat parameters
  int32_t mirostat = -1;
  float mirostat_tau = -1.0f;
  float mirostat_eta = -1.0f;

  // Other generation parameters
  int32_t n_probs = -1;
  int32_t min_keep = -1;

  // Stop sequences (optional)
  std::vector<std::string> stop_sequences;
  bool stop_sequences_set = false;

  // Grammar (optional)
  std::string grammar;
  bool grammar_set = false;

  wasi_nn_runtime_params() = default;
};

// Enhanced task structure for WASI-NN backend
struct wasi_nn_task
{
  int id = -1;
  graph_execution_context exec_ctx;
  wasi_nn_task_priority priority = WASI_NN_PRIORITY_NORMAL;
  std::chrono::steady_clock::time_point created_at;
  std::chrono::steady_clock::time_point timeout_at;
  uint32_t timeout_ms = 30000; // Default 30 second timeout
  std::string prompt;
  bool is_queued = false;

  wasi_nn_task() : created_at(std::chrono::steady_clock::now())
  {
    timeout_at = created_at + std::chrono::milliseconds(timeout_ms);
  }
};

// Forward declaration for task queue
struct wasi_nn_task_queue;

struct SessionInfo
{
  std::string session_id;
  std::vector<common_chat_msg> chat_history;
  std::chrono::steady_clock::time_point last_activity;
};

struct LlamaChatContext
{
  // Server context (from server.cpp)
  server_context server_ctx;

  // Session management (updated)
  std::unordered_map<graph_execution_context, SessionInfo> sessions;
  graph_execution_context next_exec_ctx_id;

  // Auto-cleanup configuration
  uint32_t max_sessions;
  uint32_t idle_timeout_ms;
  bool auto_cleanup_enabled;

  // Enhanced concurrency and task management (Phase 4.2)
  uint32_t queue_size;

  // Advanced task queue system
  std::shared_ptr<wasi_nn_task_queue> task_queue;
  std::thread task_processor_thread;
  bool task_processing_enabled = true;

  // Task timeout and priority settings
  uint32_t default_task_timeout_ms = 30000;
  bool priority_scheduling_enabled = true;
  bool fair_scheduling_enabled = true;

  // Queue monitoring and limits
  uint32_t queue_warning_threshold = 40;    // Warn when queue is 80% full
  uint32_t queue_reject_threshold = 50;     // Reject when queue is 100% full
  bool auto_queue_cleanup = true;

  // Memory policy
  bool context_shifting_enabled;
  std::string cache_strategy;
  uint32_t max_cache_tokens;

  // Phase 4.3: Advanced Memory Management
  uint32_t n_keep_tokens = 256;             // Number of tokens to keep when shifting context
  uint32_t n_discard_tokens = 0;            // Number of tokens to discard (0 = auto half)
  float memory_pressure_threshold = 0.85f;  // Trigger cleanup at 85% memory usage
  bool enable_partial_cache_deletion = true;
  bool enable_token_cache_reuse = true;
  std::string cache_deletion_strategy = "lru";  // lru, fifo, or smart
  uint32_t max_memory_mb = 0;               // 0 = no limit

  // Memory monitoring
  std::atomic<uint64_t> current_memory_usage{0};
  std::atomic<uint32_t> cache_hits{0};
  std::atomic<uint32_t> cache_misses{0};

  // Logging configuration
  std::string log_level;
  bool enable_debug_log;
  std::string log_file;
  bool enable_timestamps;
  bool enable_colors;

  // Logging system state
  struct common_log * log_instance;
  bool log_initialized;

  // Phase 5.2: Model Hot-Swapping
  std::string current_model_path;
  std::string current_model_version;
  bool model_swapping_in_progress;
  std::mutex model_swap_mutex;
  common_params backup_params;

  //LoRA adapters
  std::vector<common_adapter_lora_info> lora_adapters;

  // Model compatibility info
  int64_t model_context_length;
  int64_t model_vocab_size;
  std::string model_architecture;
  std::string model_name;

  // Performance settings
  bool batch_processing_enabled;
  uint32_t batch_size;

  LlamaChatContext()
      : next_exec_ctx_id(1),
        max_sessions(100), idle_timeout_ms(300000), auto_cleanup_enabled(true),
        queue_size(50),
        context_shifting_enabled(true), cache_strategy("lru"), max_cache_tokens(10000),
        n_keep_tokens(256), n_discard_tokens(0), memory_pressure_threshold(0.85f),
        enable_partial_cache_deletion(true), enable_token_cache_reuse(true),
        cache_deletion_strategy("lru"), max_memory_mb(0),
        current_memory_usage(0), cache_hits(0), cache_misses(0),
        log_level("info"), enable_debug_log(false), enable_timestamps(true), enable_colors(false),
        log_instance(nullptr), log_initialized(false),
        current_model_path(""), current_model_version(""),
        model_swapping_in_progress(false), model_context_length(0), model_vocab_size(0),
        model_architecture(""), model_name(""), lora_adapters(),
        batch_processing_enabled(true), batch_size(512) {}

  // Destructor will be defined after wasi_nn_task_queue definition
  ~LlamaChatContext();
};

// Forward declarations for helper functions
static void parse_config_to_params(const char *config_json, common_params &params, LlamaChatContext *chat_ctx = nullptr);

// Task queue with priority management
struct wasi_nn_task_queue
{
  std::deque<wasi_nn_task> high_priority_queue;    // Priority 3 (urgent)
  std::deque<wasi_nn_task> normal_priority_queue;  // Priority 1-2 (normal/high)
  std::deque<wasi_nn_task> low_priority_queue;     // Priority 0 (low)

  std::mutex queue_mutex;
  std::condition_variable queue_condition;

  uint32_t max_queue_size = 50;
  uint32_t current_size = 0;
  bool running = true;
  int next_task_id = 1;

  // Queue statistics
  uint32_t tasks_queued = 0;
  uint32_t tasks_completed = 0;
  uint32_t tasks_timeout = 0;
  uint32_t tasks_rejected = 0;

  // Add task to appropriate priority queue
  bool enqueue_task(wasi_nn_task &&task, LlamaChatContext* ctx = nullptr);

  // Get next task based on priority
  bool dequeue_task(wasi_nn_task &task, LlamaChatContext* ctx = nullptr);

  // Clean up expired tasks
  void cleanup_expired_tasks();

  // Get queue status
  void get_queue_status(uint32_t &queued, uint32_t &active, uint32_t &capacity);
};

// Implementation of LlamaChatContext destructor
LlamaChatContext::~LlamaChatContext() {
  // Cleanup logging system
  if (log_initialized && log_instance) {
    common_log_free(log_instance);
    log_instance = nullptr;
    log_initialized = false;
  }

  // Cleanup task processing thread
  if (task_processing_enabled && task_processor_thread.joinable()) {
    if (task_queue) {
      task_queue->running = false;
      task_queue->queue_condition.notify_all();
    }
    task_processor_thread.join();
  }
}

// ==============================================================================
// Phase 5.2: Stable Model Switching Implementation
// ==============================================================================

// Wait for all active tasks to complete
static wasi_nn_error wait_for_tasks_completion(LlamaChatContext *chat_ctx, uint32_t timeout_ms = 30000) {
  if (!chat_ctx || !chat_ctx->task_queue) {
    return success; // No tasks to wait for
  }

  auto start_time = std::chrono::steady_clock::now();
  auto timeout = std::chrono::milliseconds(timeout_ms);

  NN_INFO_PRINTF("Waiting for active tasks to complete before model switch...");

  while (true) {
    uint32_t queued = 0, active = 0, capacity = 0;
    chat_ctx->task_queue->get_queue_status(queued, active, capacity);

    if (active == 0 && queued == 0) {
      NN_INFO_PRINTF("All tasks completed, ready for model switch");
      return success;
    }

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    if (elapsed > timeout) {
      NN_WARN_PRINTF("Timeout waiting for tasks completion, proceeding with model switch");
      return success; // Proceed anyway after timeout
    }

    NN_DBG_PRINTF("Waiting for tasks: queued=%u, active=%u", queued, active);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// Clean up all slots and contexts before model switch
static void cleanup_all_slots(LlamaChatContext *chat_ctx) {
  if (!chat_ctx) return;

  WASI_NN_LOG_INFO(chat_ctx, "Cleaning up all slots before model switch");

  // Clear all slots using server context approach
  for (auto& slot : chat_ctx->server_ctx.slots) {
    // Free sampling context
    if (slot.smpl) {
      common_sampler_free(slot.smpl);
      slot.smpl = nullptr;
    }

    // Free draft context
    if (slot.ctx_dft) {
      llama_free(slot.ctx_dft);
      slot.ctx_dft = nullptr;
    }

    // Free speculative context
    if (slot.spec) {
      common_speculative_free(slot.spec);
      slot.spec = nullptr;
    }

    // Free batch
    if (slot.batch_spec.token) {
      llama_batch_free(slot.batch_spec);
      slot.batch_spec = {};
    }

    // Reset slot state
    slot.reset();
  }

  // Clear all slots
  chat_ctx->server_ctx.slots.clear();

  // Clear main batch
  if (chat_ctx->server_ctx.batch.token) {
    llama_batch_free(chat_ctx->server_ctx.batch);
    chat_ctx->server_ctx.batch = {};
  }

  // Clear KV cache
  if (chat_ctx->server_ctx.ctx) {
    llama_memory_t mem = llama_get_memory(chat_ctx->server_ctx.ctx);
    if (mem) {
      llama_memory_clear(mem, true);
    }
  }

  WASI_NN_LOG_INFO(chat_ctx, "All slots cleaned up successfully");
}

// Safely switch to a new model
static wasi_nn_error safe_model_switch(LlamaChatContext *chat_ctx, const char *filename,
                                       uint32_t filename_len, const char *config) {
  if (!chat_ctx) {
    return invalid_argument;
  }

  // Lock to prevent concurrent access during model switch
  std::lock_guard<std::mutex> lock(chat_ctx->model_swap_mutex);

  if (chat_ctx->model_swapping_in_progress) {
    WASI_NN_LOG_WARN(chat_ctx, "Model switch already in progress, skipping");
    return runtime_error;
  }

  chat_ctx->model_swapping_in_progress = true;

  WASI_NN_LOG_INFO(chat_ctx, "Starting safe model switch to: %.*s", (int)filename_len, filename);

  try {
    // Step 1: Wait for all active tasks to complete
    wasi_nn_error wait_result = wait_for_tasks_completion(chat_ctx, 30000);
    if (wait_result != success) {
      WASI_NN_LOG_WARN(chat_ctx, "Task completion wait failed, continuing with model switch");
    }

    // Step 2: Backup current parameters
    chat_ctx->backup_params = chat_ctx->server_ctx.params_base;

    // Step 3: Parse new configuration
    common_params new_params = chat_ctx->server_ctx.params_base;
    if (config) {
      parse_config_to_params(config, new_params, chat_ctx);
    }
    new_params.model.path = std::string(filename, filename_len);

    WASI_NN_LOG_INFO(chat_ctx, "New model config: n_gpu_layers=%d, ctx_size=%d, batch_size=%d, threads=%d",
                     new_params.n_gpu_layers, new_params.n_ctx,
                     new_params.n_batch, new_params.cpuparams.n_threads);

    // Step 4: Clean up all existing slots and contexts
    cleanup_all_slots(chat_ctx);

    // Step 5: Reset server context state
    chat_ctx->server_ctx.llama_init.model.reset();
    chat_ctx->server_ctx.llama_init.context.reset();
    chat_ctx->server_ctx.llama_init_dft.model.reset();
    chat_ctx->server_ctx.llama_init_dft.context.reset();

    chat_ctx->server_ctx.model = nullptr;
    chat_ctx->server_ctx.ctx = nullptr;
    chat_ctx->server_ctx.model_dft = nullptr;
    chat_ctx->server_ctx.vocab = nullptr;

    // Step 6: Load new model
    chat_ctx->server_ctx.params_base = new_params;

    if (!chat_ctx->server_ctx.load_model(new_params)) {
      WASI_NN_LOG_ERROR(chat_ctx, "Failed to load new model, attempting to restore previous model");

      // Attempt to restore previous model
      if (!chat_ctx->server_ctx.load_model(chat_ctx->backup_params)) {
        WASI_NN_LOG_ERROR(chat_ctx, "Failed to restore previous model - system in unstable state");
        chat_ctx->model_swapping_in_progress = false;
        return runtime_error;
      }

      WASI_NN_LOG_INFO(chat_ctx, "Previous model restored successfully");
      chat_ctx->model_swapping_in_progress = false;
      return runtime_error;
    }

    if (!chat_ctx->server_ctx.load_adapter(new_params) || !chat_ctx->server_ctx.model){
        WASI_NN_LOG_ERROR(chat_ctx, "Failed to load LoRA Adapter");
        return runtime_error;
    }

    // Step 7: Reinitialize server context
    chat_ctx->server_ctx.init();

    // Step 8: Update model information
    chat_ctx->current_model_path = std::string(filename, filename_len);
    chat_ctx->model_context_length = llama_model_n_ctx_train(chat_ctx->server_ctx.model);
    chat_ctx->model_vocab_size = llama_vocab_n_tokens(chat_ctx->server_ctx.vocab);

    // Get model architecture and name if available
    char model_desc[256] = {0};
    if (llama_model_desc(chat_ctx->server_ctx.model, model_desc, sizeof(model_desc)) > 0) {
      chat_ctx->model_architecture = std::string(model_desc);
    }

    // Extract model name from path
    std::string path(filename, filename_len);
    size_t last_slash = path.find_last_of("/\\");
    chat_ctx->model_name = (last_slash != std::string::npos) ?
                           path.substr(last_slash + 1) : path;

    // Generate version string
    struct stat file_stat;
    if (stat(filename, &file_stat) == 0) {
      char version_buf[64];
      snprintf(version_buf, sizeof(version_buf), "size_%ld_mtime_%ld",
               file_stat.st_size, file_stat.st_mtime);
      chat_ctx->current_model_version = std::string(version_buf);
    }

    // Step 9: Clear all sessions (context will be lost)
    chat_ctx->sessions.clear();
    chat_ctx->next_exec_ctx_id = 1;

    WASI_NN_LOG_INFO(chat_ctx, "Model switch completed successfully");
    WASI_NN_LOG_INFO(chat_ctx, "Model info: name=%s, arch=%s, vocab_size=%ld, ctx_len=%ld",
                     chat_ctx->model_name.c_str(), chat_ctx->model_architecture.c_str(),
                     chat_ctx->model_vocab_size, chat_ctx->model_context_length);

    chat_ctx->model_swapping_in_progress = false;
    return success;

  } catch (const std::exception& e) {
    WASI_NN_LOG_ERROR(chat_ctx, "Exception during model switch: %s", e.what());

    // Attempt to restore previous model
    try {
      if (!chat_ctx->server_ctx.load_model(chat_ctx->backup_params)) {
        WASI_NN_LOG_ERROR(chat_ctx, "Failed to restore previous model after exception");
      } else {
        WASI_NN_LOG_INFO(chat_ctx, "Previous model restored after exception");
        chat_ctx->server_ctx.init();
      }
    } catch (...) {
      WASI_NN_LOG_ERROR(chat_ctx, "Exception during model restoration");
    }

    chat_ctx->model_swapping_in_progress = false;
    return runtime_error;
  }
}

// Convert string log level to common_log verbosity level
static int string_to_log_verbosity(const std::string& level) {
  if (level == "debug" || level == "DEBUG") return 0;
  if (level == "info" || level == "INFO") return 1;
  if (level == "warn" || level == "warning" || level == "WARN" || level == "WARNING") return 2;
  if (level == "error" || level == "ERROR") return 2;
  if (level == "none" || level == "NONE" || level == "off" || level == "OFF") return 4;
  return 1; // Default to INFO level
}

// Initialize advanced logging system
static bool initialize_advanced_logging(LlamaChatContext* chat_ctx) {
  if (!chat_ctx) return false;

  // Initialize llama.cpp logging system
  if (chat_ctx->log_instance) {
    common_log_free(chat_ctx->log_instance);
  }

  chat_ctx->log_instance = common_log_init();
  if (!chat_ctx->log_instance) {
    NN_ERR_PRINTF("Failed to initialize advanced logging system");
    return false;
  }

  // Set logging verbosity based on configuration
  int verbosity = string_to_log_verbosity(chat_ctx->log_level);
  common_log_set_verbosity_thold(verbosity);

  // Configure colors
  common_log_set_colors(chat_ctx->log_instance, chat_ctx->enable_colors);

  // Configure timestamps and prefixes
  common_log_set_timestamps(chat_ctx->log_instance, chat_ctx->enable_timestamps);
  common_log_set_prefix(chat_ctx->log_instance, true);

  // Configure file output if specified
  if (!chat_ctx->log_file.empty()) {
    common_log_set_file(chat_ctx->log_instance, chat_ctx->log_file.c_str());
  }

  chat_ctx->log_initialized = true;

  // Log system initialization success
  LOG_INF("Advanced logging system initialized");
  LOG_INF("Log level: %s (verbosity: %d)", chat_ctx->log_level.c_str(), verbosity);
  LOG_INF("Debug mode: %s", chat_ctx->enable_debug_log ? "enabled" : "disabled");
  LOG_INF("Colors: %s", chat_ctx->enable_colors ? "enabled" : "disabled");
  LOG_INF("Timestamps: %s", chat_ctx->enable_timestamps ? "enabled" : "disabled");
  if (!chat_ctx->log_file.empty()) {
    LOG_INF("File logging: %s", chat_ctx->log_file.c_str());
  }

  return true;
}

// Structured logging for task queue operations
static void log_task_operation(LlamaChatContext* chat_ctx, const std::string& operation,
                              int task_id, wasi_nn_task_priority priority = WASI_NN_PRIORITY_NORMAL,
                              const std::string& additional_info = "") {
  if (!chat_ctx || !chat_ctx->log_initialized) return;

  const char* priority_str = "NORMAL";
  switch (priority) {
    case WASI_NN_PRIORITY_LOW: priority_str = "LOW"; break;
    case WASI_NN_PRIORITY_NORMAL: priority_str = "NORMAL"; break;
    case WASI_NN_PRIORITY_HIGH: priority_str = "HIGH"; break;
    case WASI_NN_PRIORITY_URGENT: priority_str = "URGENT"; break;
  }

  if (additional_info.empty()) {
    LOG_INF("[TASK] %s - Task %d (Priority: %s)", operation.c_str(), task_id, priority_str);
  } else {
    LOG_INF("[TASK] %s - Task %d (Priority: %s) - %s",
            operation.c_str(), task_id, priority_str, additional_info.c_str());
  }
}

// Implementation of wasi_nn_task_queue methods
bool wasi_nn_task_queue::enqueue_task(wasi_nn_task &&task, LlamaChatContext* ctx)
{
  std::unique_lock<std::mutex> lock(queue_mutex);

  // Check if queue is at capacity
  if (current_size >= max_queue_size) {
    tasks_rejected++;
    if (ctx) {
      WASI_NN_LOG_WARN(ctx, "Task queue full (%d/%d), rejecting task %d",
                       current_size, max_queue_size, task.id);
    } else {
      NN_WARN_PRINTF("Task queue full (%d/%d), rejecting task %d",
                     current_size, max_queue_size, task.id);
    }
    return false;
  }

  // Assign task ID if not set
  if (task.id == -1) {
    task.id = next_task_id++;
  }

  // Add to appropriate priority queue
  switch (task.priority) {
    case WASI_NN_PRIORITY_URGENT:
      high_priority_queue.push_back(std::move(task));
      break;
    case WASI_NN_PRIORITY_HIGH:
    case WASI_NN_PRIORITY_NORMAL:
      normal_priority_queue.push_back(std::move(task));
      break;
    case WASI_NN_PRIORITY_LOW:
    default:
      low_priority_queue.push_back(std::move(task));
      break;
  }

  current_size++;
  tasks_queued++;

  // Use advanced logging if available
  if (ctx) {
    log_task_operation(ctx, "Task Queued", task.id, task.priority,
                      "Queue: " + std::to_string(current_size) + "/" + std::to_string(max_queue_size));
  } else {
    NN_INFO_PRINTF("Task %d queued with priority %d. Queue size: %d/%d",
                   task.id, (int)task.priority, current_size, max_queue_size);
  }

  // Notify waiting threads
  queue_condition.notify_one();
  return true;
}

bool wasi_nn_task_queue::dequeue_task(wasi_nn_task &task, LlamaChatContext* ctx)
{
  std::unique_lock<std::mutex> lock(queue_mutex);

  // Wait for tasks to become available
  queue_condition.wait(lock, [this] {
    return !running ||
           !high_priority_queue.empty() ||
           !normal_priority_queue.empty() ||
           !low_priority_queue.empty();
  });

  if (!running) {
    return false;
  }

  // Clean up expired tasks first
  cleanup_expired_tasks();

  // Dequeue from highest priority queue first
  if (!high_priority_queue.empty()) {
    task = std::move(high_priority_queue.front());
    high_priority_queue.pop_front();
  } else if (!normal_priority_queue.empty()) {
    task = std::move(normal_priority_queue.front());
    normal_priority_queue.pop_front();
  } else if (!low_priority_queue.empty()) {
    task = std::move(low_priority_queue.front());
    low_priority_queue.pop_front();
  } else {
    return false; // No tasks available
  }

  current_size--;

  // Use advanced logging if available
  if (ctx) {
    log_task_operation(ctx, "Task Dequeued", task.id, task.priority,
                      "Queue: " + std::to_string(current_size) + "/" + std::to_string(max_queue_size));
  } else {
    NN_INFO_PRINTF("Dequeued task %d with priority %d. Queue size: %d/%d",
                   task.id, (int)task.priority, current_size, max_queue_size);
  }

  return true;
}

void wasi_nn_task_queue::cleanup_expired_tasks()
{
  // Note: This method assumes the queue_mutex is already locked
  auto now = std::chrono::steady_clock::now();

  auto cleanup_queue = [&](std::deque<wasi_nn_task> &queue) {
    auto it = queue.begin();
    while (it != queue.end()) {
      if (now > it->timeout_at) {
        NN_WARN_PRINTF("Task %d expired (created %ldms ago)",
                       it->id,
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - it->created_at).count());
        it = queue.erase(it);
        current_size--;
        tasks_timeout++;
      } else {
        ++it;
      }
    }
  };

  cleanup_queue(high_priority_queue);
  cleanup_queue(normal_priority_queue);
  cleanup_queue(low_priority_queue);
}

void wasi_nn_task_queue::get_queue_status(uint32_t &queued, uint32_t &active, uint32_t &capacity)
{
  std::unique_lock<std::mutex> lock(queue_mutex);
  queued = current_size;
  active = tasks_queued - tasks_completed - tasks_timeout - tasks_rejected;
  capacity = max_queue_size;
}

// Phase 4.3: Advanced Memory Management Functions
// ================================================

// Memory monitoring and pressure detection
static uint64_t get_current_memory_usage() {
  // Simple implementation using /proc/self/status on Linux
  FILE* file = fopen("/proc/self/status", "r");
  if (!file) {
    return 0;
  }

  char line[256];
  uint64_t rss_kb = 0;

  while (fgets(line, sizeof(line), file)) {
    if (sscanf(line, "VmRSS: %lu kB", &rss_kb) == 1) {
      break;
    }
  }

  fclose(file);
  return rss_kb * 1024; // Convert to bytes
}

static bool check_memory_pressure(LlamaChatContext* chat_ctx) {
  if (chat_ctx->max_memory_mb == 0) {
    return false; // No memory limit set
  }

  uint64_t current_mb = chat_ctx->current_memory_usage.load() / (1024 * 1024);
  uint64_t max_mb = chat_ctx->max_memory_mb;
  float usage_ratio = (float)current_mb / max_mb;

  return usage_ratio >= chat_ctx->memory_pressure_threshold;
}

// Context shifting implementation based on server.cpp
static wasi_nn_error perform_context_shift(LlamaChatContext* chat_ctx, uint32_t session_id) {
  if (!chat_ctx->context_shifting_enabled) {
    NN_ERR_PRINTF("Context shifting is disabled");
    return runtime_error;
  }

  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;

  if (!ctx) {
    NN_ERR_PRINTF("No context available for shifting");
    return runtime_error;
  }

  const int n_ctx = llama_n_ctx(ctx);
  const int n_keep = chat_ctx->n_keep_tokens;

  // In server.cpp, n_past is tracked per slot - here we use a simplified approach
  // For a full implementation, you would track n_past per session
  int n_past = n_ctx * 0.8f; // Assume 80% filled as a simplified estimate
  const int n_left = n_past - n_keep;

  if (n_left <= 0) {
    NN_WARN_PRINTF("No tokens to shift (n_past=%d, n_keep=%d)", n_past, n_keep);
    return success;
  }

  const int n_discard = chat_ctx->n_discard_tokens > 0 ?
                        chat_ctx->n_discard_tokens : (n_left / 2);

  NN_INFO_PRINTF("Performing context shift: n_keep=%d, n_left=%d, n_discard=%d",
                 n_keep, n_left, n_discard);

  // Perform the actual context shift using llama.cpp memory functions
  llama_memory_seq_rm(llama_get_memory(ctx), session_id, n_keep, n_keep + n_discard);
  llama_memory_seq_add(llama_get_memory(ctx), session_id, n_keep + n_discard, n_past, -n_discard);

  NN_INFO_PRINTF("Context shift completed successfully");
  return success;
}

// Partial KV cache deletion strategies
static wasi_nn_error clear_partial_kv_cache(LlamaChatContext* chat_ctx, uint32_t session_id,
                                           const std::string& strategy) {
  if (!chat_ctx->enable_partial_cache_deletion) {
    NN_WARN_PRINTF("Partial cache deletion is disabled");
    return invalid_argument;
  }

  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;

  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache deletion");
    return runtime_error;
  }

  const int n_ctx = llama_n_ctx(ctx);
  // Simplified approach - estimate current usage as 80% of context size
  const int n_past = n_ctx * 0.8f;

  if (strategy == "lru") {
    // Clear the oldest entries (simplified implementation)
    const int n_clear = n_past / 4; // Clear 25% of oldest entries

    if (n_clear > 0) {
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, 0, n_clear);
      NN_INFO_PRINTF("Cleared %d oldest KV cache entries using LRU strategy", n_clear);
    }
  } else if (strategy == "fifo") {
    // Clear the newest entries
    const int n_clear = n_past / 4;

    if (n_clear > 0) {
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, n_past - n_clear, n_past);
      NN_INFO_PRINTF("Cleared %d newest KV cache entries using FIFO strategy", n_clear);
    }
  } else if (strategy == "smart") {
    // Smart deletion based on token importance (simplified)
    const int n_keep = chat_ctx->n_keep_tokens;
    const int n_clear = (n_past - n_keep) / 2;

    if (n_clear > 0) {
      // Keep important tokens at the beginning and end, clear middle
      const int clear_start = n_keep + n_clear / 2;
      llama_memory_seq_rm(llama_get_memory(ctx), session_id, clear_start, clear_start + n_clear);
      NN_INFO_PRINTF("Cleared %d middle KV cache entries using smart strategy", n_clear);
    }
  } else {
    NN_ERR_PRINTF("Unknown cache deletion strategy: %s", strategy.c_str());
    return invalid_argument;
  }

  return success;
}

// Token cache reuse mechanism
static wasi_nn_error optimize_token_cache(LlamaChatContext* chat_ctx, uint32_t session_id) {
  if (!chat_ctx->enable_token_cache_reuse) {
    return success; // Not enabled, but not an error
  }

  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;

  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache optimization");
    return runtime_error;
  }

  const int n_ctx = llama_n_ctx(ctx);
  // Simplified approach - estimate cached tokens
  const int n_cached = n_ctx * 0.7f; // Assume 70% cached

  if (n_cached > (int)chat_ctx->max_cache_tokens) {
    // Perform cache cleanup
    wasi_nn_error result = clear_partial_kv_cache(chat_ctx, session_id,
                                                  chat_ctx->cache_deletion_strategy);
    if (result != success) {
      NN_WARN_PRINTF("Failed to optimize token cache: %d", result);
      return result;
    }

    chat_ctx->cache_hits++;
    NN_INFO_PRINTF("Token cache optimized: %d tokens cached, hit ratio: %.2f%%",
                    n_cached,
                    (float)chat_ctx->cache_hits.load() /
                    (chat_ctx->cache_hits.load() + chat_ctx->cache_misses.load()) * 100.0f);
  } else {
    chat_ctx->cache_misses++;
  }

  return success;
}

// Complete KV cache clear (based on server.cpp implementation)
static wasi_nn_error clear_kv_cache(LlamaChatContext* chat_ctx, uint32_t session_id) {
  auto& server_ctx = chat_ctx->server_ctx;
  llama_context* ctx = server_ctx.ctx;

  if (!ctx) {
    NN_ERR_PRINTF("No context available for cache clearing");
    return runtime_error;
  }

  NN_INFO_PRINTF("Clearing KV cache for session %u", session_id);

  if (session_id == 0) {
    // Clear entire KV cache
    llama_memory_clear(llama_get_memory(ctx), true);
    NN_INFO_PRINTF("Cleared entire KV cache");
  } else {
    // Clear cache for specific session
    llama_memory_seq_rm(llama_get_memory(ctx), session_id, -1, -1);
    NN_INFO_PRINTF("Cleared KV cache for session %u", session_id);
  }

  return success;
}

// Memory pressure handling
static wasi_nn_error handle_memory_pressure(LlamaChatContext* chat_ctx) {
  NN_WARN_PRINTF("Memory pressure detected, initiating cleanup");

  // Strategy 1: Clear partial caches for all active sessions
  wasi_nn_error result = clear_partial_kv_cache(chat_ctx, 0, chat_ctx->cache_deletion_strategy);
  if (result != success) {
    NN_WARN_PRINTF("Partial cache cleanup failed, trying full cache clear");

    // Strategy 2: Clear entire cache if partial cleanup failed
    result = clear_kv_cache(chat_ctx, 0);
    if (result != success) {
      NN_ERR_PRINTF("Failed to handle memory pressure");
      return result;
    }
  }

  // Update memory tracking
  chat_ctx->current_memory_usage.store(get_current_memory_usage());

  NN_INFO_PRINTF("Memory pressure handling completed");
  return success;
}

// Enhanced helper function for safe JSON value extraction (similar to server.cpp json_value)
template <typename T>
static T cjson_get_value(cJSON *root, const char *key, const T &default_value);

// Specializations for different types
template <>
double cjson_get_value<double>(cJSON *root, const char *key, const double &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsNumber(item))
  {
    return cJSON_GetNumberValue(item);
  }
  return default_value;
}

template <>
float cjson_get_value<float>(cJSON *root, const char *key, const float &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsNumber(item))
  {
    return (float)cJSON_GetNumberValue(item);
  }
  return default_value;
}

template <>
int32_t cjson_get_value<int32_t>(cJSON *root, const char *key, const int32_t &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsNumber(item))
  {
    return (int32_t)cJSON_GetNumberValue(item);
  }
  return default_value;
}

template <>
uint32_t cjson_get_value<uint32_t>(cJSON *root, const char *key, const uint32_t &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsNumber(item))
  {
    return (uint32_t)cJSON_GetNumberValue(item);
  }
  return default_value;
}

template <>
bool cjson_get_value<bool>(cJSON *root, const char *key, const bool &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsBool(item))
  {
    return cJSON_IsTrue(item);
  }
  return default_value;
}

template <>
std::string cjson_get_value<std::string>(cJSON *root, const char *key, const std::string &default_value)
{
  cJSON *item = cJSON_GetObjectItem(root, key);
  if (cJSON_IsString(item))
  {
    return std::string(cJSON_GetStringValue(item));
  }
  return default_value;
}

// Function to parse runtime parameters from JSON configuration
static bool parse_runtime_params(const char *config_json, uint32_t config_len,
                                wasi_nn_runtime_params &runtime_params,
                                LlamaChatContext *chat_ctx = nullptr)
{
  if (!config_json || config_len == 0) {
    if (chat_ctx) {
      WASI_NN_LOG_INFO(chat_ctx, "No runtime config provided, using defaults");
    }
    return true; // Not an error, just use defaults
  }

  cJSON *root = cJSON_ParseWithLength(config_json, config_len);
  if (!root) {
    if (chat_ctx) {
      WASI_NN_LOG_ERROR(chat_ctx, "Failed to parse runtime configuration JSON");
    }
    return false;
  }

  // Parse core sampling parameters
  runtime_params.temperature = cjson_get_value(root, "temperature", runtime_params.temperature);
  runtime_params.temperature = cjson_get_value(root, "temp", runtime_params.temperature); // Alternative name
  runtime_params.top_p = cjson_get_value(root, "top_p", runtime_params.top_p);
  runtime_params.top_k = cjson_get_value(root, "top_k", runtime_params.top_k);
  runtime_params.min_p = cjson_get_value(root, "min_p", runtime_params.min_p);
  runtime_params.typical_p = cjson_get_value(root, "typical_p", runtime_params.typical_p);

  // Parse penalty parameters
  runtime_params.repeat_penalty = cjson_get_value(root, "repeat_penalty", runtime_params.repeat_penalty);
  runtime_params.frequency_penalty = cjson_get_value(root, "frequency_penalty", runtime_params.frequency_penalty);
  runtime_params.presence_penalty = cjson_get_value(root, "presence_penalty", runtime_params.presence_penalty);
  runtime_params.penalty_last_n = cjson_get_value(root, "penalty_last_n", runtime_params.penalty_last_n);
  runtime_params.penalty_last_n = cjson_get_value(root, "repeat_last_n", runtime_params.penalty_last_n); // OpenAI compatibility

  // Parse generation control parameters
  runtime_params.max_tokens = cjson_get_value(root, "max_tokens", runtime_params.max_tokens);
  runtime_params.max_tokens = cjson_get_value(root, "n_predict", runtime_params.max_tokens); // Alternative name
  runtime_params.seed = cjson_get_value(root, "seed", 1234); // Default seed

  // Parse ignore_eos with explicit flag
  cJSON *ignore_eos_item = cJSON_GetObjectItem(root, "ignore_eos");
  if (cJSON_IsBool(ignore_eos_item)) {
    runtime_params.ignore_eos = cJSON_IsTrue(ignore_eos_item);
    runtime_params.ignore_eos_set = true;
  }

  // Parse DRY sampling parameters
  runtime_params.dry_multiplier = cjson_get_value(root, "dry_multiplier", runtime_params.dry_multiplier);
  runtime_params.dry_base = cjson_get_value(root, "dry_base", runtime_params.dry_base);
  runtime_params.dry_allowed_length = cjson_get_value(root, "dry_allowed_length", runtime_params.dry_allowed_length);
  runtime_params.dry_penalty_last_n = cjson_get_value(root, "dry_penalty_last_n", runtime_params.dry_penalty_last_n);

  // Parse dynamic temperature parameters
  runtime_params.dynatemp_range = cjson_get_value(root, "dynatemp_range", runtime_params.dynatemp_range);
  runtime_params.dynatemp_exponent = cjson_get_value(root, "dynatemp_exponent", runtime_params.dynatemp_exponent);

  // Parse Mirostat parameters
  runtime_params.mirostat = cjson_get_value(root, "mirostat", runtime_params.mirostat);
  runtime_params.mirostat_tau = cjson_get_value(root, "mirostat_tau", runtime_params.mirostat_tau);
  runtime_params.mirostat_eta = cjson_get_value(root, "mirostat_eta", runtime_params.mirostat_eta);

  // Parse other parameters
  runtime_params.n_probs = cjson_get_value(root, "n_probs", runtime_params.n_probs);
  runtime_params.n_probs = cjson_get_value(root, "logprobs", runtime_params.n_probs); // OpenAI compatibility
  runtime_params.min_keep = cjson_get_value(root, "min_keep", runtime_params.min_keep);

  // Parse stop sequences
  cJSON *stop = cJSON_GetObjectItem(root, "stop");
  if (cJSON_IsArray(stop)) {
    runtime_params.stop_sequences.clear();
    int array_size = cJSON_GetArraySize(stop);
    for (int i = 0; i < array_size; i++) {
      cJSON *stop_item = cJSON_GetArrayItem(stop, i);
      if (cJSON_IsString(stop_item)) {
        std::string stop_word = cJSON_GetStringValue(stop_item);
        if (!stop_word.empty()) {
          runtime_params.stop_sequences.push_back(stop_word);
        }
      }
    }
    runtime_params.stop_sequences_set = true;
  }

  // Parse grammar
  cJSON *grammar_item = cJSON_GetObjectItem(root, "grammar");
  if (cJSON_IsString(grammar_item)) {
    runtime_params.grammar = cJSON_GetStringValue(grammar_item);
    runtime_params.grammar_set = true;
  }

  // Parameter validation
  if (runtime_params.temperature > 0.0f && (runtime_params.temperature < 0.01f || runtime_params.temperature > 10.0f)) {
    if (chat_ctx) {
      WASI_NN_LOG_WARN(chat_ctx, "Temperature %.3f out of reasonable range [0.01, 10.0], using as-is", runtime_params.temperature);
    }
  }

  if (runtime_params.top_p > 0.0f && (runtime_params.top_p < 0.01f || runtime_params.top_p > 1.0f)) {
    if (chat_ctx) {
      WASI_NN_LOG_WARN(chat_ctx, "top_p %.3f out of valid range [0.01, 1.0], clamping", runtime_params.top_p);
    }
    runtime_params.top_p = std::max(0.01f, std::min(1.0f, runtime_params.top_p));
  }

  if (runtime_params.repeat_penalty > 0.0f && runtime_params.repeat_penalty < 0.1f) {
    if (chat_ctx) {
      WASI_NN_LOG_WARN(chat_ctx, "repeat_penalty %.3f too low, setting to 0.1", runtime_params.repeat_penalty);
    }
    runtime_params.repeat_penalty = 0.1f;
  }

  cJSON_Delete(root);

  if (chat_ctx) {
    WASI_NN_LOG_INFO(chat_ctx, "Runtime parameters parsed successfully");
  }

  return true;
}

// Function to apply runtime parameters to sampling context
static void apply_runtime_params_to_sampling(common_sampler *&sampler,
                                            const wasi_nn_runtime_params &runtime_params,
                                            const llama_model *model,
                                            LlamaChatContext *chat_ctx = nullptr)
{
  if (!sampler || !model) {
    if (chat_ctx) {
      WASI_NN_LOG_ERROR(chat_ctx, "Invalid sampler context or model for runtime parameter application");
    }
    return;
  }

  // Get current sampling parameters from the chat context
  common_params_sampling current_params = chat_ctx->server_ctx.params_base.sampling;
  bool params_changed = false;

  // Apply core sampling parameters
  if (runtime_params.temperature >= 0.0f) {
    current_params.temp = runtime_params.temperature;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied temperature: %.3f", runtime_params.temperature);
    }
  }

  if (runtime_params.top_p >= 0.0f) {
    current_params.top_p = runtime_params.top_p;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied top_p: %.3f", runtime_params.top_p);
    }
  }

  if (runtime_params.top_k >= 0) {
    current_params.top_k = runtime_params.top_k;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied top_k: %d", runtime_params.top_k);
    }
  }

  if (runtime_params.min_p >= 0.0f) {
    current_params.min_p = runtime_params.min_p;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied min_p: %.3f", runtime_params.min_p);
    }
  }

  if (runtime_params.typical_p >= 0.0f) {
    current_params.typ_p = runtime_params.typical_p;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied typical_p: %.3f", runtime_params.typical_p);
    }
  }

  // Apply penalty parameters
  if (runtime_params.repeat_penalty >= 0.0f) {
    current_params.penalty_repeat = runtime_params.repeat_penalty;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied repeat_penalty: %.3f", runtime_params.repeat_penalty);
    }
  }

  if (runtime_params.frequency_penalty >= 0.0f) {
    current_params.penalty_freq = runtime_params.frequency_penalty;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied frequency_penalty: %.3f", runtime_params.frequency_penalty);
    }
  }

  if (runtime_params.presence_penalty >= 0.0f) {
    current_params.penalty_present = runtime_params.presence_penalty;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied presence_penalty: %.3f", runtime_params.presence_penalty);
    }
  }

  if (runtime_params.penalty_last_n >= 0) {
    current_params.penalty_last_n = runtime_params.penalty_last_n;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied penalty_last_n: %d", runtime_params.penalty_last_n);
    }
  }

  // Apply DRY sampling parameters
  if (runtime_params.dry_multiplier >= 0.0f) {
    current_params.dry_multiplier = runtime_params.dry_multiplier;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dry_multiplier: %.3f", runtime_params.dry_multiplier);
    }
  }

  if (runtime_params.dry_base >= 0.0f) {
    current_params.dry_base = runtime_params.dry_base;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dry_base: %.3f", runtime_params.dry_base);
    }
  }

  if (runtime_params.dry_allowed_length >= 0) {
    current_params.dry_allowed_length = runtime_params.dry_allowed_length;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dry_allowed_length: %d", runtime_params.dry_allowed_length);
    }
  }

  if (runtime_params.dry_penalty_last_n >= 0) {
    current_params.dry_penalty_last_n = runtime_params.dry_penalty_last_n;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dry_penalty_last_n: %d", runtime_params.dry_penalty_last_n);
    }
  }

  // Apply dynamic temperature parameters
  if (runtime_params.dynatemp_range >= 0.0f) {
    current_params.dynatemp_range = runtime_params.dynatemp_range;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dynatemp_range: %.3f", runtime_params.dynatemp_range);
    }
  }

  if (runtime_params.dynatemp_exponent >= 0.0f) {
    current_params.dynatemp_exponent = runtime_params.dynatemp_exponent;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied dynatemp_exponent: %.3f", runtime_params.dynatemp_exponent);
    }
  }

  // Apply Mirostat parameters
  if (runtime_params.mirostat >= 0) {
    current_params.mirostat = runtime_params.mirostat;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied mirostat: %d", runtime_params.mirostat);
    }
  }

  if (runtime_params.mirostat_tau >= 0.0f) {
    current_params.mirostat_tau = runtime_params.mirostat_tau;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied mirostat_tau: %.3f", runtime_params.mirostat_tau);
    }
  }

  if (runtime_params.mirostat_eta >= 0.0f) {
    current_params.mirostat_eta = runtime_params.mirostat_eta;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied mirostat_eta: %.3f", runtime_params.mirostat_eta);
    }
  }

  // Apply other parameters
  if (runtime_params.seed >= 0) {
    current_params.seed = runtime_params.seed;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied seed: %d", runtime_params.seed);
    }
  }

  if (runtime_params.n_probs >= 0) {
    current_params.n_probs = runtime_params.n_probs;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied n_probs: %d", runtime_params.n_probs);
    }
  }

  if (runtime_params.min_keep >= 0) {
    current_params.min_keep = runtime_params.min_keep;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied min_keep: %d", runtime_params.min_keep);
    }
  }

  if (runtime_params.ignore_eos_set) {
    current_params.ignore_eos = runtime_params.ignore_eos;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied ignore_eos: %s", runtime_params.ignore_eos ? "true" : "false");
    }
  }

  // Apply grammar if provided
  if (runtime_params.grammar_set && !runtime_params.grammar.empty()) {
    current_params.grammar = runtime_params.grammar;
    params_changed = true;
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "Applied grammar: %s", runtime_params.grammar.c_str());
    }
  }

  // If any parameters changed, recreate the sampler
  if (params_changed) {
    // Free the old sampler
    common_sampler_free(sampler);

    // Create new sampler with updated parameters
    sampler = common_sampler_init(model, current_params);

    if (!sampler) {
      if (chat_ctx) {
        WASI_NN_LOG_ERROR(chat_ctx, "Failed to recreate sampler with runtime parameters");
      }
      return;
    }

    if (chat_ctx) {
      WASI_NN_LOG_INFO(chat_ctx, "Runtime parameters applied to sampler successfully - sampler recreated");
    }
  } else {
    if (chat_ctx) {
      WASI_NN_LOG_DEBUG(chat_ctx, "No runtime parameters provided or changed, using existing sampler");
    }
  }
}

// Enhanced parameter parsing function (based on server.cpp params_from_json_cmpl)
static void parse_config_to_params(const char *config_json,
                                   common_params &params,
                                   LlamaChatContext *chat_ctx)
{
  // Initialize with sensible defaults (server.cpp style)
  params = common_params();
  params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
  params.enable_chat_template = true;

  // Model defaults
  params.n_predict = 512;
  params.n_ctx = 2048;
  params.n_batch = 512;
  params.n_gpu_layers = 0;
  params.cpuparams.n_threads = 8;
  params.cpuparams_batch.n_threads = 8;

  // Sampling defaults (matching server.cpp defaults)
  params.sampling.temp = 0.7f;
  params.sampling.top_p = 0.95f;
  params.sampling.top_k = -1;
  params.sampling.min_p = 0.0f;
  params.sampling.typ_p = 1.0f;
  params.sampling.penalty_repeat = 1.10f;
  params.sampling.penalty_freq = 0.0f;
  params.sampling.penalty_present = 0.0f;
  params.sampling.penalty_last_n = -1;  // Will be auto-adjusted
  params.sampling.ignore_eos = false;
  params.sampling.seed = 1234; // Default seed for reproducibility
  params.sampling.n_probs = 0;
  params.sampling.min_keep = 1;

  // DRY sampling defaults
  params.sampling.dry_multiplier = 0.0f;
  params.sampling.dry_base = 1.75f;
  params.sampling.dry_allowed_length = 2;
  params.sampling.dry_penalty_last_n = -1;  // Will be auto-adjusted

  // Dynatemp defaults
  params.sampling.dynatemp_range = 0.0f;
  params.sampling.dynatemp_exponent = 1.0f;

  // Mirostat defaults
  params.sampling.mirostat = 0;
  params.sampling.mirostat_tau = 5.0f;
  params.sampling.mirostat_eta = 0.1f;

  if (!config_json)
  {
    if (chat_ctx) {
      WASI_NN_LOG_INFO(chat_ctx, "No configuration provided, using defaults");
    }
    return;
  }

  cJSON *root = cJSON_Parse(config_json);
  if (!root)
  {
    if (chat_ctx) {
      WASI_NN_LOG_ERROR(chat_ctx, "Failed to parse configuration JSON");
    }
    return;
  }

  // Parse model parameters with comprehensive error handling
  auto parse_model_params = [&](cJSON *config_obj) {
    params.n_predict = cjson_get_value(config_obj, "n_predict", params.n_predict);
    params.n_predict = cjson_get_value(config_obj, "max_tokens", params.n_predict);  // OpenAI compatibility
    params.n_gpu_layers = cjson_get_value(config_obj, "n_gpu_layers", params.n_gpu_layers);
    params.n_ctx = cjson_get_value(config_obj, "ctx_size", params.n_ctx);
    params.n_ctx = cjson_get_value(config_obj, "n_ctx", params.n_ctx);  // Alternative name
    params.n_batch = cjson_get_value(config_obj, "batch_size", params.n_batch);
    params.n_batch = cjson_get_value(config_obj, "n_batch", params.n_batch);  // Alternative name

    uint32_t threads = cjson_get_value(config_obj, "threads", params.cpuparams.n_threads);
    params.cpuparams.n_threads = threads;
    params.cpuparams_batch.n_threads = threads;
  };

  // Parse nested model configuration or legacy flat structure
  cJSON *model_config = cJSON_GetObjectItem(root, "model");
  if (cJSON_IsObject(model_config))
  {
    parse_model_params(model_config);
  }
  else
  {
    // Legacy flat configuration (backward compatibility)
    parse_model_params(root);
  }


  cJSON *lora_array = cJSON_GetObjectItem(root, "lora_adapters");
  if (cJSON_IsArray(lora_array)) {
      params.lora_adapters.clear();
      cJSON *lora_item;
      cJSON_ArrayForEach(lora_item, lora_array) {
          cJSON *path = cJSON_GetObjectItem(lora_item, "path");
          cJSON *scale = cJSON_GetObjectItem(lora_item, "scale");

          if (cJSON_IsString(path) && cJSON_IsNumber(scale)) {
              common_adapter_lora_info adapter;
              adapter.path = cJSON_GetStringValue(path);
              adapter.scale = (float)cJSON_GetNumberValue(scale);
              adapter.ptr = nullptr; // Will be set during loading
              params.lora_adapters.push_back(adapter);
          }
      }
  }
  // Parse sampling parameters - Legacy flat structure first (backward compatibility)
  params.sampling.temp = cjson_get_value(root, "temp", params.sampling.temp);
  params.sampling.temp = cjson_get_value(root, "temperature", params.sampling.temp);  // OpenAI compatibility
  params.sampling.top_p = cjson_get_value(root, "top_p", params.sampling.top_p);
  params.sampling.penalty_repeat = cjson_get_value(root, "repeat_penalty", params.sampling.penalty_repeat);

  // Parse nested sampling configuration (server.cpp style)
  cJSON *sampling = cJSON_GetObjectItem(root, "sampling");
  if (cJSON_IsObject(sampling))
  {
    // Core sampling parameters
    params.sampling.temp = cjson_get_value(sampling, "temp", params.sampling.temp);
    params.sampling.temp = cjson_get_value(sampling, "temperature", params.sampling.temp);
    params.sampling.top_p = cjson_get_value(sampling, "top_p", params.sampling.top_p);
    params.sampling.top_k = cjson_get_value(sampling, "top_k", params.sampling.top_k);
    params.sampling.min_p = cjson_get_value(sampling, "min_p", params.sampling.min_p);
    params.sampling.typ_p = cjson_get_value(sampling, "typical_p", params.sampling.typ_p);

    // Penalty parameters
    params.sampling.penalty_repeat = cjson_get_value(sampling, "repeat_penalty", params.sampling.penalty_repeat);
    params.sampling.penalty_present = cjson_get_value(sampling, "presence_penalty", params.sampling.penalty_present);
    params.sampling.penalty_freq = cjson_get_value(sampling, "frequency_penalty", params.sampling.penalty_freq);
    params.sampling.penalty_last_n = cjson_get_value(sampling, "penalty_last_n", params.sampling.penalty_last_n);
    params.sampling.penalty_last_n = cjson_get_value(sampling, "repeat_last_n", params.sampling.penalty_last_n);  // OpenAI compatibility

    // DRY sampling parameters (advanced repetition suppression)
    params.sampling.dry_multiplier = cjson_get_value(sampling, "dry_multiplier", params.sampling.dry_multiplier);
    params.sampling.dry_base = cjson_get_value(sampling, "dry_base", params.sampling.dry_base);
    params.sampling.dry_allowed_length = cjson_get_value(sampling, "dry_allowed_length", params.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = cjson_get_value(sampling, "dry_penalty_last_n", params.sampling.dry_penalty_last_n);

    // Dynamic temperature parameters
    params.sampling.dynatemp_range = cjson_get_value(sampling, "dynatemp_range", params.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent = cjson_get_value(sampling, "dynatemp_exponent", params.sampling.dynatemp_exponent);

    // Mirostat parameters
    params.sampling.mirostat = cjson_get_value(sampling, "mirostat", params.sampling.mirostat);
    params.sampling.mirostat_tau = cjson_get_value(sampling, "mirostat_tau", params.sampling.mirostat_tau);
    params.sampling.mirostat_eta = cjson_get_value(sampling, "mirostat_eta", params.sampling.mirostat_eta);

    // Other sampling parameters
    params.sampling.seed = cjson_get_value(sampling, "seed", params.sampling.seed);
    params.sampling.n_probs = cjson_get_value(sampling, "n_probs", params.sampling.n_probs);
    params.sampling.n_probs = cjson_get_value(sampling, "logprobs", params.sampling.n_probs);  // OpenAI compatibility
    params.sampling.min_keep = cjson_get_value(sampling, "min_keep", params.sampling.min_keep);
    params.sampling.ignore_eos = cjson_get_value(sampling, "ignore_eos", params.sampling.ignore_eos);

    // Grammar parameters
    params.sampling.grammar = cjson_get_value(sampling, "grammar", params.sampling.grammar);
    params.sampling.grammar_lazy = cjson_get_value(sampling, "grammar_lazy", params.sampling.grammar_lazy);

    // DRY sequence breakers (server.cpp style)
    cJSON *dry_sequence_breakers = cJSON_GetObjectItem(sampling, "dry_sequence_breakers");
    if (cJSON_IsArray(dry_sequence_breakers))
    {
      params.sampling.dry_sequence_breakers.clear();
      int array_size = cJSON_GetArraySize(dry_sequence_breakers);
      for (int i = 0; i < array_size; i++)
      {
        cJSON *breaker_item = cJSON_GetArrayItem(dry_sequence_breakers, i);
        if (cJSON_IsString(breaker_item))
        {
          params.sampling.dry_sequence_breakers.push_back(std::string(cJSON_GetStringValue(breaker_item)));
        }
      }

      if (params.sampling.dry_sequence_breakers.empty())
      {
        if (chat_ctx) {
          WASI_NN_LOG_ERROR(chat_ctx, "Error: dry_sequence_breakers must be a non-empty array of strings");
        }
        cJSON_Delete(root);
        return;
      }
    }
  }

  // Parse stopping criteria (enhanced version)
  cJSON *stopping = cJSON_GetObjectItem(root, "stopping");
  if (cJSON_IsObject(stopping))
  {
    params.n_predict = cjson_get_value(stopping, "max_tokens", params.n_predict);
    params.sampling.ignore_eos = cjson_get_value(stopping, "ignore_eos", params.sampling.ignore_eos);

    // Parse stop sequences (server.cpp style)
    cJSON *stop = cJSON_GetObjectItem(stopping, "stop");
    if (cJSON_IsArray(stop))
    {
      params.antiprompt.clear();
      int array_size = cJSON_GetArraySize(stop);
      for (int i = 0; i < array_size; i++)
      {
        cJSON *stop_item = cJSON_GetArrayItem(stop, i);
        if (cJSON_IsString(stop_item))
        {
          std::string stop_word = cJSON_GetStringValue(stop_item);
          if (!stop_word.empty())
          {
            params.antiprompt.push_back(stop_word);
          }
        }
      }
    }
  }

  // Parse logit bias (server.cpp style)
  cJSON *logit_bias = cJSON_GetObjectItem(root, "logit_bias");
  if (cJSON_IsArray(logit_bias))
  {
    params.sampling.logit_bias.clear();
    int array_size = cJSON_GetArraySize(logit_bias);
    for (int i = 0; i < array_size; i++)
    {
      cJSON *bias_item = cJSON_GetArrayItem(logit_bias, i);
      if (cJSON_IsArray(bias_item) && cJSON_GetArraySize(bias_item) == 2)
      {
        cJSON *token_item = cJSON_GetArrayItem(bias_item, 0);
        cJSON *bias_value = cJSON_GetArrayItem(bias_item, 1);

        if (cJSON_IsNumber(token_item) && cJSON_IsNumber(bias_value))
        {
          llama_token token = (llama_token)cJSON_GetNumberValue(token_item);
          float bias = (float)cJSON_GetNumberValue(bias_value);
          params.sampling.logit_bias.push_back({token, bias});
        }
      }
    }
  }

  // Critical parameter validation (based on server.cpp params_from_json_cmpl)
  if (params.sampling.penalty_last_n < -1)
  {
    if (chat_ctx) {
      WASI_NN_LOG_ERROR(chat_ctx, "Error: repeat_last_n must be >= -1");
    }
    cJSON_Delete(root);
    return;
  }

  if (params.sampling.dry_penalty_last_n < -1)
  {
    if (chat_ctx) {
      WASI_NN_LOG_ERROR(chat_ctx, "Error: dry_penalty_last_n must be >= -1");
    }
    cJSON_Delete(root);
    return;
  }

  // Auto-adjust -1 values to context size (simplified, no ctx available here)
  if (params.sampling.penalty_last_n == -1)
  {
    params.sampling.penalty_last_n = params.n_ctx;
  }

  if (params.sampling.dry_penalty_last_n == -1)
  {
    params.sampling.dry_penalty_last_n = params.n_ctx;
  }

  // Validate DRY base parameter
  if (params.sampling.dry_base < 1.0f)
  {
    if (chat_ctx) {
      WASI_NN_LOG_WARN(chat_ctx, "dry_base (%.3f) < 1.0, resetting to default (%.3f)",
                       params.sampling.dry_base, 1.75f);
    }
    params.sampling.dry_base = 1.75f;
  }

  cJSON_Delete(root);

  if (chat_ctx) {
    WASI_NN_LOG_INFO(chat_ctx, "Configuration parsed successfully");
  }
}

// Phase 4.3: Parse advanced memory management configuration (optimized)
static void parse_memory_config(const char *config_json, LlamaChatContext *chat_ctx)
{
  if (!config_json || !chat_ctx)
    return;

  cJSON *root = cJSON_Parse(config_json);
  if (!root)
  {
    WASI_NN_LOG_WARN(chat_ctx, "Failed to parse config JSON for memory settings");
    return;
  }

  cJSON *memory = cJSON_GetObjectItem(root, "memory");
  if (cJSON_IsObject(memory))
  {
    // Context shifting settings
    chat_ctx->context_shifting_enabled = cjson_get_value(memory, "context_shifting", chat_ctx->context_shifting_enabled);

    // Cache strategy with validation
    std::string cache_strategy = cjson_get_value(memory, "cache_strategy", chat_ctx->cache_strategy);
    if (cache_strategy == "lru" || cache_strategy == "fifo" || cache_strategy == "smart")
    {
      chat_ctx->cache_strategy = cache_strategy;
      WASI_NN_LOG_INFO(chat_ctx, "Cache strategy set to: %s", cache_strategy.c_str());
    }
    else if (!cache_strategy.empty() && cache_strategy != chat_ctx->cache_strategy)
    {
      WASI_NN_LOG_WARN(chat_ctx, "Invalid cache strategy '%s', using default '%s'",
                       cache_strategy.c_str(), chat_ctx->cache_strategy.c_str());
    }

    // Maximum cache tokens with validation
    uint32_t max_cache_tokens = cjson_get_value(memory, "max_cache_tokens", chat_ctx->max_cache_tokens);
    if (max_cache_tokens > 0)
    {
      chat_ctx->max_cache_tokens = max_cache_tokens;
      WASI_NN_LOG_INFO(chat_ctx, "Max cache tokens set to: %u", max_cache_tokens);
    }
    else if (max_cache_tokens == 0)
    {
      WASI_NN_LOG_WARN(chat_ctx, "max_cache_tokens cannot be 0, using default: %u", chat_ctx->max_cache_tokens);
    }

    // Keep tokens with validation
    uint32_t n_keep_tokens = cjson_get_value(memory, "n_keep_tokens", chat_ctx->n_keep_tokens);
    if (n_keep_tokens <= 4096)  // Reasonable upper limit
    {
      chat_ctx->n_keep_tokens = n_keep_tokens;
      WASI_NN_LOG_INFO(chat_ctx, "Keep tokens set to: %u", n_keep_tokens);
    }
    else
    {
      WASI_NN_LOG_WARN(chat_ctx, "n_keep_tokens (%u) too large, using default: %u",
                       n_keep_tokens, chat_ctx->n_keep_tokens);
    }

    // Discard tokens
    chat_ctx->n_discard_tokens = cjson_get_value(memory, "n_discard_tokens", chat_ctx->n_discard_tokens);

    // Memory pressure threshold with validation
    float memory_pressure_threshold = cjson_get_value(memory, "memory_pressure_threshold", chat_ctx->memory_pressure_threshold);
    if (memory_pressure_threshold >= 0.1f && memory_pressure_threshold <= 1.0f)
    {
      chat_ctx->memory_pressure_threshold = memory_pressure_threshold;
      WASI_NN_LOG_INFO(chat_ctx, "Memory pressure threshold set to: %.2f", memory_pressure_threshold);
    }
    else
    {
      WASI_NN_LOG_WARN(chat_ctx, "Invalid memory_pressure_threshold (%.2f), must be between 0.1 and 1.0, using default: %.2f",
                       memory_pressure_threshold, chat_ctx->memory_pressure_threshold);
    }

    // Boolean settings
    chat_ctx->enable_partial_cache_deletion = cjson_get_value(memory, "enable_partial_cache_deletion", chat_ctx->enable_partial_cache_deletion);
    chat_ctx->enable_token_cache_reuse = cjson_get_value(memory, "enable_token_cache_reuse", chat_ctx->enable_token_cache_reuse);

    // Cache deletion strategy with validation
    std::string cache_deletion_strategy = cjson_get_value(memory, "cache_deletion_strategy", chat_ctx->cache_deletion_strategy);
    if (cache_deletion_strategy == "lru" || cache_deletion_strategy == "fifo" || cache_deletion_strategy == "smart")
    {
      chat_ctx->cache_deletion_strategy = cache_deletion_strategy;
      WASI_NN_LOG_INFO(chat_ctx, "Cache deletion strategy set to: %s", cache_deletion_strategy.c_str());
    }
    else if (!cache_deletion_strategy.empty() && cache_deletion_strategy != chat_ctx->cache_deletion_strategy)
    {
      WASI_NN_LOG_WARN(chat_ctx, "Invalid cache deletion strategy '%s', using default '%s'",
                       cache_deletion_strategy.c_str(), chat_ctx->cache_deletion_strategy.c_str());
    }

    // Memory limit with validation
    uint32_t max_memory_mb = cjson_get_value(memory, "max_memory_mb", chat_ctx->max_memory_mb);
    if (max_memory_mb == 0 || max_memory_mb >= 64)  // 0 = unlimited, or at least 64MB
    {
      chat_ctx->max_memory_mb = max_memory_mb;
      if (max_memory_mb == 0)
      {
        WASI_NN_LOG_INFO(chat_ctx, "Memory limit disabled (unlimited)");
      }
      else
      {
        WASI_NN_LOG_INFO(chat_ctx, "Max memory limit set to: %u MB", max_memory_mb);
      }
    }
    else
    {
      WASI_NN_LOG_WARN(chat_ctx, "max_memory_mb (%u) too small, minimum is 64MB, using default: %u",
                       max_memory_mb, chat_ctx->max_memory_mb);
    }

    WASI_NN_LOG_INFO(chat_ctx, "Memory configuration parsed successfully");
  }
  else if (cJSON_GetObjectItem(root, "memory"))
  {
    WASI_NN_LOG_WARN(chat_ctx, "Memory configuration is not a valid object");
  }

  cJSON_Delete(root);
}

// Helper function to setup threadpools (from main.cpp)
static wasi_nn_error setup_threadpools(LlamaChatContext *chat_ctx)
{
  // Access server context members through server_ctx
  common_params& params = chat_ctx->server_ctx.params_base;

  auto *reg = ggml_backend_dev_backend_reg(
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
  auto ggml_threadpool_new_fn =
      (decltype(ggml_threadpool_new) *)ggml_backend_reg_get_proc_address(
          reg, "ggml_threadpool_new");

  struct ggml_threadpool_params tpp_batch =
      ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
  struct ggml_threadpool_params tpp =
      ggml_threadpool_params_from_cpu_params(params.cpuparams);

  // Create batch threadpool if different from main threadpool
  ggml_threadpool* threadpool_batch = nullptr;
  if (!ggml_threadpool_params_match(&tpp, &tpp_batch))
  {
    threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
    if (!threadpool_batch)
    {
      NN_ERR_PRINTF("Failed to create batch threadpool");
      return runtime_error;
    }
    tpp.paused = true;
  }

  ggml_threadpool* threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool)
  {
    NN_ERR_PRINTF("Failed to create threadpool");
    return runtime_error;
  }

  llama_attach_threadpool(chat_ctx->server_ctx.ctx, threadpool,
                          threadpool_batch);
  return success;
}

// ===============================================
// Phase 4.3: Forward declarations for internal memory management functions
// ===============================================
static wasi_nn_error auto_clear_kv_cache_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);
static wasi_nn_error auto_clear_all_kv_cache(LlamaChatContext *chat_ctx);
static wasi_nn_error auto_perform_context_shift_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);
static wasi_nn_error auto_optimize_memory(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx);

// Function to safely copy a string into tensor_data (from original)
void copy_string_to_tensor_data(tensor_data dest, uint32_t dest_size,
                                const std::string &src)
{
  if (dest == nullptr || dest_size == 0)
  {
    NN_ERR_PRINTF("Destination buffer is null or size is zero");
    return;
  }

  size_t src_size = src.size();
  if (src_size >= dest_size)
  {
    NN_WARN_PRINTF("Source string is too long, truncating");
    src_size = dest_size - 1;
  }

  std::copy(src.begin(), src.begin() + src_size,
            reinterpret_cast<uint8_t *>(dest));
  if (dest_size > 0)
  {
    dest[src_size] = '\0';
  }
}

// Main API functions
__attribute__((visibility("default"))) wasi_nn_error init_backend(void **ctx)
{
  return init_backend_with_config(ctx, nullptr, 0);
}

__attribute__((visibility("default"))) wasi_nn_error
init_backend_with_config(void **ctx, const char *config, uint32_t config_len)
{
    if (!ctx) {
      NN_ERR_PRINTF("The context pointer provided to init_backend cannot be null.");
      return invalid_argument;
    }
  LlamaChatContext *chat_ctx = new LlamaChatContext();
  if (!chat_ctx)
  {
    NN_ERR_PRINTF("Failed to allocate chat context");
    return runtime_error;
  }

  // Parse config JSON to update settings if provided
  if (config && config_len > 0)
  {
    cJSON *json = cJSON_ParseWithLength(config, config_len);
    if (!json) {
         NN_ERR_PRINTF("Failed to parse configuration JSON. The provided string is not valid.");
         delete chat_ctx; // Clean up the allocated context
         *ctx = NULL;
         return invalid_argument; // Return an error to stop initialization
     }

    if (json)
    {
      // Helper function to parse backend configuration (optimized)
      auto parse_backend_config = [&](cJSON *config_obj) {
        // Session management settings with validation
        uint32_t max_sessions = cjson_get_value(config_obj, "max_sessions", chat_ctx->max_sessions);
        if (max_sessions > 0 && max_sessions <= 10000)  // Reasonable range
        {
          chat_ctx->max_sessions = max_sessions;
          WASI_NN_LOG_INFO(chat_ctx, "Max sessions set to: %u", max_sessions);
        }
        else if (max_sessions != chat_ctx->max_sessions)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid max_sessions (%u), using default: %u",
                           max_sessions, chat_ctx->max_sessions);
        }

        // Timeout settings with validation
        uint32_t idle_timeout = cjson_get_value(config_obj, "idle_timeout_ms", chat_ctx->idle_timeout_ms);
        if (idle_timeout >= 1000 && idle_timeout <= 86400000)  // 1s to 24h
        {
          chat_ctx->idle_timeout_ms = idle_timeout;
          WASI_NN_LOG_INFO(chat_ctx, "Idle timeout set to: %u ms", idle_timeout);
        }
        else if (idle_timeout != chat_ctx->idle_timeout_ms)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid idle_timeout_ms (%u), must be between 1000-86400000, using default: %u",
                           idle_timeout, chat_ctx->idle_timeout_ms);
        }

        // Boolean settings
        chat_ctx->auto_cleanup_enabled = cjson_get_value(config_obj, "auto_cleanup", chat_ctx->auto_cleanup_enabled);

        // Queue size with validation
        uint32_t queue_size = cjson_get_value(config_obj, "queue_size", chat_ctx->queue_size);
        if (queue_size > 0 && queue_size <= 10000)  // Reasonable range
        {
          chat_ctx->queue_size = queue_size;
          WASI_NN_LOG_INFO(chat_ctx, "Queue size set to: %u", queue_size);

          // Auto-adjust thresholds based on queue size
          chat_ctx->queue_warning_threshold = std::min(chat_ctx->queue_warning_threshold,
                                                       static_cast<uint32_t>(queue_size * 0.8f));
          chat_ctx->queue_reject_threshold = queue_size;
        }
        else if (queue_size != chat_ctx->queue_size)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid queue_size (%u), must be between 1-10000, using default: %u",
                           queue_size, chat_ctx->queue_size);
        }

        // Task timeout with validation
        uint32_t task_timeout = cjson_get_value(config_obj, "default_task_timeout_ms", chat_ctx->default_task_timeout_ms);
        if (task_timeout >= 1000 && task_timeout <= 600000)  // 1s to 10min
        {
          chat_ctx->default_task_timeout_ms = task_timeout;
          WASI_NN_LOG_INFO(chat_ctx, "Default task timeout set to: %u ms", task_timeout);
        }
        else if (task_timeout != chat_ctx->default_task_timeout_ms)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid default_task_timeout_ms (%u), must be between 1000-600000, using default: %u",
                           task_timeout, chat_ctx->default_task_timeout_ms);
        }

        // Boolean scheduling settings
        chat_ctx->priority_scheduling_enabled = cjson_get_value(config_obj, "priority_scheduling_enabled",
                                                               chat_ctx->priority_scheduling_enabled);
        chat_ctx->fair_scheduling_enabled = cjson_get_value(config_obj, "fair_scheduling_enabled",
                                                           chat_ctx->fair_scheduling_enabled);
        chat_ctx->auto_queue_cleanup = cjson_get_value(config_obj, "auto_queue_cleanup",
                                                      chat_ctx->auto_queue_cleanup);

        // Queue threshold settings with validation
        uint32_t queue_warning = cjson_get_value(config_obj, "queue_warning_threshold", chat_ctx->queue_warning_threshold);
        uint32_t queue_reject = cjson_get_value(config_obj, "queue_reject_threshold", chat_ctx->queue_reject_threshold);

        if (queue_warning <= chat_ctx->queue_size && queue_reject <= chat_ctx->queue_size &&
            queue_warning <= queue_reject)
        {
          chat_ctx->queue_warning_threshold = queue_warning;
          chat_ctx->queue_reject_threshold = queue_reject;
          WASI_NN_LOG_INFO(chat_ctx, "Queue thresholds: warning=%u, reject=%u", queue_warning, queue_reject);
        }
        else
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid queue thresholds (warning=%u, reject=%u), using defaults: warning=%u, reject=%u",
                           queue_warning, queue_reject, chat_ctx->queue_warning_threshold, chat_ctx->queue_reject_threshold);
        }
      };

      // Parse backend configuration - first check for new nested structure
      cJSON *backend_config = cJSON_GetObjectItem(json, "backend");
      if (cJSON_IsObject(backend_config))
      {
        // New nested backend configuration
        parse_backend_config(backend_config);
        WASI_NN_LOG_INFO(chat_ctx, "Loaded nested backend configuration");
      }
      else
      {
        // Legacy flat configuration (backward compatibility)
        parse_backend_config(json);
        WASI_NN_LOG_INFO(chat_ctx, "Loaded flat backend configuration (legacy mode)");
      }

      // Memory policy with enhanced parsing
      cJSON *memory_policy = cJSON_GetObjectItem(json, "memory_policy");
      if (cJSON_IsObject(memory_policy))
      {
        chat_ctx->context_shifting_enabled = cjson_get_value(memory_policy, "context_shifting",
                                                            chat_ctx->context_shifting_enabled);

        std::string cache_strategy = cjson_get_value(memory_policy, "cache_strategy", chat_ctx->cache_strategy);
        if (cache_strategy == "lru" || cache_strategy == "fifo" || cache_strategy == "smart")
        {
          chat_ctx->cache_strategy = cache_strategy;
          WASI_NN_LOG_INFO(chat_ctx, "Memory cache strategy set to: %s", cache_strategy.c_str());
        }
        else if (!cache_strategy.empty() && cache_strategy != chat_ctx->cache_strategy)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid memory cache strategy '%s', using default '%s'",
                           cache_strategy.c_str(), chat_ctx->cache_strategy.c_str());
        }

        uint32_t max_cache_tokens = cjson_get_value(memory_policy, "max_cache_tokens", chat_ctx->max_cache_tokens);
        if (max_cache_tokens >= 1024 && max_cache_tokens <= 1000000)  // 1K to 1M tokens
        {
          chat_ctx->max_cache_tokens = max_cache_tokens;
          WASI_NN_LOG_INFO(chat_ctx, "Max cache tokens set to: %u", max_cache_tokens);
        }
        else if (max_cache_tokens != chat_ctx->max_cache_tokens)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid max_cache_tokens (%u), must be between 1024-1000000, using default: %u",
                           max_cache_tokens, chat_ctx->max_cache_tokens);
        }

        uint32_t max_memory_mb = cjson_get_value(memory_policy, "max_memory_mb", chat_ctx->max_memory_mb);
        if (max_memory_mb == 0 || (max_memory_mb >= 128 && max_memory_mb <= 32768))  // 0=unlimited, 128MB to 32GB
        {
          chat_ctx->max_memory_mb = max_memory_mb;
          WASI_NN_LOG_INFO(chat_ctx, "Max memory limit set to: %u MB (0=unlimited)", max_memory_mb);
        }
        else if (max_memory_mb != chat_ctx->max_memory_mb)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid max_memory_mb (%u), must be 0 or between 128-32768, using default: %u",
                           max_memory_mb, chat_ctx->max_memory_mb);
        }

        // Memory pressure threshold
        float memory_pressure = cjson_get_value(memory_policy, "memory_pressure_threshold", chat_ctx->memory_pressure_threshold);
        if (memory_pressure >= 0.5f && memory_pressure <= 0.95f)
        {
          chat_ctx->memory_pressure_threshold = memory_pressure;
          WASI_NN_LOG_INFO(chat_ctx, "Memory pressure threshold set to: %.2f", memory_pressure);
        }
        else if (memory_pressure != chat_ctx->memory_pressure_threshold)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid memory_pressure_threshold (%.2f), must be between 0.5-0.95, using default: %.2f",
                           memory_pressure, chat_ctx->memory_pressure_threshold);
        }

        // Token keep/discard settings
        uint32_t n_keep_tokens = cjson_get_value(memory_policy, "n_keep_tokens", chat_ctx->n_keep_tokens);
        if (n_keep_tokens >= 64 && n_keep_tokens <= 2048)
        {
          chat_ctx->n_keep_tokens = n_keep_tokens;
          WASI_NN_LOG_INFO(chat_ctx, "Keep tokens set to: %u", n_keep_tokens);
        }
        else if (n_keep_tokens != chat_ctx->n_keep_tokens)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid n_keep_tokens (%u), must be between 64-2048, using default: %u",
                           n_keep_tokens, chat_ctx->n_keep_tokens);
        }

        // Boolean memory settings
        chat_ctx->enable_partial_cache_deletion = cjson_get_value(memory_policy, "enable_partial_cache_deletion",
                                                                 chat_ctx->enable_partial_cache_deletion);
        chat_ctx->enable_token_cache_reuse = cjson_get_value(memory_policy, "enable_token_cache_reuse",
                                                            chat_ctx->enable_token_cache_reuse);

        // Cache deletion strategy
        std::string cache_delete_strategy = cjson_get_value(memory_policy, "cache_deletion_strategy", chat_ctx->cache_deletion_strategy);
        if (cache_delete_strategy == "lru" || cache_delete_strategy == "fifo" || cache_delete_strategy == "smart")
        {
          chat_ctx->cache_deletion_strategy = cache_delete_strategy;
          WASI_NN_LOG_INFO(chat_ctx, "Cache deletion strategy set to: %s", cache_delete_strategy.c_str());
        }
        else if (!cache_delete_strategy.empty() && cache_delete_strategy != chat_ctx->cache_deletion_strategy)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid cache_deletion_strategy '%s', using default '%s'",
                           cache_delete_strategy.c_str(), chat_ctx->cache_deletion_strategy.c_str());
        }
      }

      // Logging configuration with enhanced validation
      cJSON *logging = cJSON_GetObjectItem(json, "logging");
      if (cJSON_IsObject(logging))
      {
        std::string log_level = cjson_get_value(logging, "level", chat_ctx->log_level);
        if (log_level == "debug" || log_level == "info" || log_level == "warn" ||
            log_level == "error" || log_level == "fatal")
        {
          chat_ctx->log_level = log_level;
          WASI_NN_LOG_INFO(chat_ctx, "Log level set to: %s", log_level.c_str());
        }
        else if (!log_level.empty() && log_level != chat_ctx->log_level)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid log level '%s', using default '%s'",
                           log_level.c_str(), chat_ctx->log_level.c_str());
        }

        // Boolean logging settings
        chat_ctx->enable_debug_log = cjson_get_value(logging, "enable_debug", chat_ctx->enable_debug_log);
        chat_ctx->enable_timestamps = cjson_get_value(logging, "timestamps", chat_ctx->enable_timestamps);
        chat_ctx->enable_colors = cjson_get_value(logging, "colors", chat_ctx->enable_colors);

        // Log file path validation
        std::string log_file = cjson_get_value(logging, "file", chat_ctx->log_file);
        if (!log_file.empty())
        {
          chat_ctx->log_file = log_file;
          WASI_NN_LOG_INFO(chat_ctx, "Log file set to: %s", log_file.c_str());
        }
      }      // Performance settings with validation
      cJSON *performance = cJSON_GetObjectItem(json, "performance");
      if (cJSON_IsObject(performance))
      {
        chat_ctx->batch_processing_enabled = cjson_get_value(performance, "batch_processing",
                                                            chat_ctx->batch_processing_enabled);

        uint32_t batch_size = cjson_get_value(performance, "batch_size", chat_ctx->batch_size);
        if (batch_size >= 1 && batch_size <= 2048)  // Reasonable batch size range
        {
          chat_ctx->batch_size = batch_size;
          WASI_NN_LOG_INFO(chat_ctx, "Batch size set to: %u", batch_size);
        }
        else if (batch_size != chat_ctx->batch_size)
        {
          WASI_NN_LOG_WARN(chat_ctx, "Invalid batch_size (%u), must be between 1-2048, using default: %u",
                           batch_size, chat_ctx->batch_size);
        }
      }

      cJSON *lora_array = cJSON_GetObjectItem(json, "lora_adapters");
        if (cJSON_IsArray(lora_array)) {
            cJSON *lora_item;
            cJSON_ArrayForEach(lora_item, lora_array) {
                cJSON *path = cJSON_GetObjectItem(lora_item, "path");
                cJSON *scale = cJSON_GetObjectItem(lora_item, "scale");
                WASI_NN_LOG_INFO(chat_ctx, "the path of LoRA adapter is %s", path->valuestring);
                WASI_NN_LOG_INFO(chat_ctx, "the path of scale is %f", scale->valuedouble);
                }
            }




      cJSON_Delete(json);
    }

    // Phase 4.3: Parse advanced memory management settings
    if (config && config_len > 0) {
      std::string config_str(config, config_len);
      parse_memory_config(config_str.c_str(), chat_ctx);
    }
  }

  // Initialize llama backend (from main.cpp)
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  // Initialize task queue system (Phase 4.2)
  chat_ctx->task_queue = std::make_shared<wasi_nn_task_queue>();
  chat_ctx->task_queue->max_queue_size = chat_ctx->queue_size;

  // Start task processing thread if enabled
  if (chat_ctx->task_processing_enabled) {
    chat_ctx->task_processor_thread = std::thread([chat_ctx]() {
      NN_INFO_PRINTF("Task processor thread started");

      wasi_nn_task task;
      while (chat_ctx->task_queue->running) {
        if (chat_ctx->task_queue->dequeue_task(task, chat_ctx)) {
          // Process the task
          NN_INFO_PRINTF("Processing task %d for execution context %d",
                         task.id, task.exec_ctx);

          // For now, just mark as completed
          // In a full implementation, this would trigger actual inference
          {
            std::unique_lock<std::mutex> lock(chat_ctx->task_queue->queue_mutex);
            chat_ctx->task_queue->tasks_completed++;
          }

          NN_INFO_PRINTF("Task %d completed", task.id);
        }
      }

      NN_INFO_PRINTF("Task processor thread terminated");
    });
  }

  NN_INFO_PRINTF("Llama chat backend initialized successfully");

  // Phase 5.1: Initialize advanced logging system
  initialize_advanced_logging(chat_ctx);

  // Use enhanced logging for configuration output
  WASI_NN_LOG_INFO(chat_ctx,
      "Session config: max_sessions=%d, idle_timeout_ms=%d, auto_cleanup=%s",
      chat_ctx->max_sessions, chat_ctx->idle_timeout_ms,
      chat_ctx->auto_cleanup_enabled ? "true" : "false");
  WASI_NN_LOG_INFO(chat_ctx,
      "Queue config: queue_size=%d",
      chat_ctx->queue_size);
  WASI_NN_LOG_INFO(chat_ctx,
      "Task Queue config: timeout=%dms, priority_scheduling=%s, fair_scheduling=%s",
      chat_ctx->default_task_timeout_ms,
      chat_ctx->priority_scheduling_enabled ? "true" : "false",
      chat_ctx->fair_scheduling_enabled ? "true" : "false");
  WASI_NN_LOG_INFO(chat_ctx,
      "Memory config: context_shifting=%s, cache_strategy=%s, max_cache_tokens=%d",
      chat_ctx->context_shifting_enabled ? "true" : "false",
      chat_ctx->cache_strategy.c_str(), chat_ctx->max_cache_tokens);
  WASI_NN_LOG_INFO(chat_ctx,
      "Logging config: level=%s, enable_debug=%s, timestamps=%s, colors=%s, file=%s",
      chat_ctx->log_level.c_str(),
      chat_ctx->enable_debug_log ? "true" : "false",
      chat_ctx->enable_timestamps ? "true" : "false",
      chat_ctx->enable_colors ? "true" : "false",
      chat_ctx->log_file.c_str());
  WASI_NN_LOG_INFO(chat_ctx,
      "Performance config: batch_processing=%s, batch_size=%d",
      chat_ctx->batch_processing_enabled ? "true" : "false",
      chat_ctx->batch_size);
  *ctx = (void *)chat_ctx;
  return success;
}

__attribute__((visibility("default"))) wasi_nn_error deinit_backend(void *ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  // Note: model and ctx are managed by common_init_result's unique_ptrs
  // They will be automatically cleaned up by the server_context

  llama_backend_free();
  delete chat_ctx;
  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name_with_config(void *ctx, const char *filename, uint32_t filename_len,
                         const char *config, uint32_t config_len, graph *g)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  if (!filename) {
    NN_ERR_PRINTF("The model filename cannot be null.");
    return invalid_argument;
  }

  NN_DBG_PRINTF("Loading model: %s", filename);
  NN_DBG_PRINTF("Config: %s", config ? config : "null");

  // Check if this is a model switch (if a model is already loaded)
  bool is_model_switch = (chat_ctx->server_ctx.model != nullptr);

  if (is_model_switch) {
    NN_INFO_PRINTF("Performing safe model switch from %s to %s",
                   chat_ctx->current_model_path.c_str(), filename);

    // Use safe model switching
    wasi_nn_error switch_result = safe_model_switch(chat_ctx, filename, filename_len, config);
    if (switch_result != success) {
      NN_ERR_PRINTF("Safe model switch failed: %d", switch_result);
      return switch_result;
    }

    NN_INFO_PRINTF("Safe model switch completed successfully");
    return success;
  }

  // Initial model loading (no existing model)
  // Parse config into params
  parse_config_to_params(config, chat_ctx->server_ctx.params_base, chat_ctx);
  chat_ctx->server_ctx.params_base.model.path = filename;

  NN_INFO_PRINTF("Model config: n_gpu_layers=%d, ctx_size=%d, batch_size=%d, threads=%d",
                 chat_ctx->server_ctx.params_base.n_gpu_layers,
                 chat_ctx->server_ctx.params_base.n_ctx,
                 chat_ctx->server_ctx.params_base.n_batch,
                 chat_ctx->server_ctx.params_base.cpuparams.n_threads);

  // Load model using server_context's approach
  if (!chat_ctx->server_ctx.load_model(chat_ctx->server_ctx.params_base)) {
      NN_ERR_PRINTF("Failed to load model from file %s", filename);
      return runtime_error;
  }

  // Initialize server context
  chat_ctx->server_ctx.init();

  // Check context size
  const int n_ctx_train = llama_model_n_ctx_train(chat_ctx->server_ctx.model);
  const int n_ctx = llama_n_ctx(chat_ctx->server_ctx.ctx);

  if (n_ctx > n_ctx_train)
  {
    NN_WARN_PRINTF("Model was trained on only %d context tokens (%d specified)",
                   n_ctx_train, n_ctx);
  }

  // Phase 5.2: Record model information for safe switching
  chat_ctx->current_model_path = std::string(filename, filename_len);
  chat_ctx->model_context_length = n_ctx_train;
  chat_ctx->model_vocab_size = llama_vocab_n_tokens(chat_ctx->server_ctx.vocab);

  // Get model architecture and name if available
  char model_desc[256] = {0};
  if (llama_model_desc(chat_ctx->server_ctx.model, model_desc, sizeof(model_desc)) > 0) {
    chat_ctx->model_architecture = std::string(model_desc);
  }

  // Extract model name from path
  std::string path(filename, filename_len);
  size_t last_slash = path.find_last_of("/\\");
  chat_ctx->model_name = (last_slash != std::string::npos) ?
                         path.substr(last_slash + 1) : path;

  // Generate simple version string based on file size and modification time
  struct stat file_stat;
  if (stat(filename, &file_stat) == 0) {
    char version_buf[64];
    snprintf(version_buf, sizeof(version_buf), "size_%ld_mtime_%ld",
             file_stat.st_size, file_stat.st_mtime);
    chat_ctx->current_model_version = std::string(version_buf);
  }

  NN_INFO_PRINTF("Model loaded successfully. Context size: %d", n_ctx);
  NN_INFO_PRINTF("Model info recorded: name=%s, arch=%s, vocab_size=%ld, ctx_len=%ld",
                 chat_ctx->model_name.c_str(), chat_ctx->model_architecture.c_str(),
                 chat_ctx->model_vocab_size, chat_ctx->model_context_length);

  return success;
}

// Auto-cleanup function: removes old/excess sessions
static void auto_cleanup_sessions(LlamaChatContext *chat_ctx)
{
  if (!chat_ctx->auto_cleanup_enabled)
    return;

  auto now = std::chrono::steady_clock::now();
  auto idle_timeout = std::chrono::milliseconds(chat_ctx->idle_timeout_ms);

  // Remove idle sessions
  for (auto it = chat_ctx->sessions.begin(); it != chat_ctx->sessions.end();)
  {
    if ((now - it->second.last_activity) > idle_timeout)
    {
      auto idle_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - it->second.last_activity)
                         .count();
      NN_INFO_PRINTF("Auto-cleanup: removing idle session %d (idle for %lldms)",
                     it->first, (long long)idle_time);
      it = chat_ctx->sessions.erase(it);
    }
    else
    {
      ++it;
    }
  }

  // Remove excess sessions (LRU eviction)
  if (chat_ctx->sessions.size() >= chat_ctx->max_sessions)
  {
    // Sort by last_activity and remove the oldest
    std::vector<std::pair<graph_execution_context,
                          std::chrono::steady_clock::time_point>>
        sorted_sessions;
    for (const auto &session : chat_ctx->sessions)
    {
      sorted_sessions.emplace_back(session.first, session.second.last_activity);
    }

    std::sort(sorted_sessions.begin(), sorted_sessions.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    // Remove oldest sessions to make room
    size_t sessions_to_remove =
        chat_ctx->sessions.size() - chat_ctx->max_sessions + 1;
    for (size_t i = 0; i < sessions_to_remove && i < sorted_sessions.size();
         ++i)
    {
      auto exec_ctx_id = sorted_sessions[i].first;
      NN_INFO_PRINTF("Auto-cleanup: removing session %d (max sessions reached)",
                     exec_ctx_id);
      chat_ctx->sessions.erase(exec_ctx_id);
    }
  }
}

// Original function for WASI-NN compatibility (kept for backward compatibility)
__attribute__((visibility("default"))) wasi_nn_error init_execution_context(
    void *ctx, graph g, graph_execution_context *exec_ctx)
{
  // Delegate to the session-aware version with a default session ID
  return init_execution_context_with_session_id(ctx, "default_session", exec_ctx);
}

// New function that properly handles session IDs (matches NIF expectations)
__attribute__((visibility("default"))) wasi_nn_error init_execution_context_with_session_id(
    void *ctx, const char *session_id, graph_execution_context *exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !chat_ctx->server_ctx.model)
    return invalid_argument;

  if (!session_id) {
    NN_ERR_PRINTF("Session ID cannot be null");
    return invalid_argument;
  }

  std::string session_id_str(session_id);

  // Check if session already exists
  for (auto &pair : chat_ctx->sessions) {
    if (pair.second.session_id == session_id_str) {
      *exec_ctx = pair.first;
      NN_INFO_PRINTF("Reusing existing session '%s' with execution context %d",
                     session_id, pair.first);
      return success;
    }
  }

  // Auto-cleanup on entry to free up space before checking session limits
  auto_cleanup_sessions(chat_ctx);

  // Check if we can create a new session (use sessions.size() vs max_sessions)
  if (chat_ctx->sessions.size() >= chat_ctx->max_sessions)
  {
    NN_ERR_PRINTF("Unable to create new session after cleanup. Current: %zu, Max: %d",
                  chat_ctx->sessions.size(), chat_ctx->max_sessions);
    return runtime_error;
  }

  // Initialize sampler if not already done (from main.cpp)
  // Setup threadpools if not already done
  wasi_nn_error result = setup_threadpools(chat_ctx);
  if (result != success)
  {
    return result;
  }

  // Initialize the server context
  chat_ctx->server_ctx.init();

  // Initialize samplers for all slots (crucial for inference)
  for (auto &slot : chat_ctx->server_ctx.slots) {
    if (slot.smpl != nullptr) {
      common_sampler_free(slot.smpl);
    }
    slot.smpl = common_sampler_init(chat_ctx->server_ctx.model, slot.params.sampling);
    if (slot.smpl == nullptr) {
      NN_ERR_PRINTF("Failed to initialize sampler for slot %d", slot.id);
      return runtime_error;
    }
  }

  // Create new session with provided session ID
  graph_execution_context new_exec_ctx = chat_ctx->next_exec_ctx_id++;
  SessionInfo session_info;
  session_info.session_id = session_id_str;  // Use the provided session ID
  session_info.last_activity = std::chrono::steady_clock::now();

  chat_ctx->sessions[new_exec_ctx] = std::move(session_info);

  *exec_ctx = new_exec_ctx;

  NN_INFO_PRINTF(
      "Execution context %d initialized for session '%s'. Total sessions: %zu, Max sessions: %d",
      new_exec_ctx, session_id, chat_ctx->sessions.size(), chat_ctx->max_sessions);

  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
close_execution_context(void *ctx, graph_execution_context exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  auto it = chat_ctx->sessions.find(exec_ctx);
  if (it != chat_ctx->sessions.end())
  {
    NN_INFO_PRINTF("Closing execution context %d for session '%s'", exec_ctx,
                   it->second.session_id.c_str());

    // Phase 4.3: Auto-clear KV cache for this session before closing
    auto_clear_kv_cache_session(chat_ctx, exec_ctx);

    chat_ctx->sessions.erase(it);

    // Phase 4.3: Check if we should do global memory optimization after session close
    if (chat_ctx->sessions.empty()) {
      // All sessions closed, good time for global cleanup
      auto_clear_all_kv_cache(chat_ctx);
    }

    return success;
  }

  return invalid_argument;
}

// Helper function to run inference loop (extracted from main.cpp)
static std::string run_inference_for_session(LlamaChatContext *chat_ctx,
                                             graph_execution_context exec_ctx,
                                             const std::string &user_input)
{
  // Find session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
  {
    return "Error: Invalid session";
  }

  SessionInfo &session_info = session_it->second;
  auto &chat_msgs = session_info.chat_history;

  // Update last activity
  session_info.last_activity = std::chrono::steady_clock::now();

  // Chat formatting function (from main.cpp)
  auto chat_add_and_format = [&](const std::string &role,
                                 const std::string &content)
  {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;

    // Check if chat templates are available
    if (!chat_ctx->server_ctx.chat_templates.get()) {
      NN_ERR_PRINTF("Chat templates not initialized");
      return std::string("Error: Chat templates not available");
    }

    auto formatted = common_chat_format_single(
        chat_ctx->server_ctx.chat_templates.get(), chat_msgs, new_msg, role == "user",
        false // use_jinja
    );

    chat_msgs.push_back(new_msg);
    NN_DBG_PRINTF("Formatted message: '%s'", formatted.c_str());
    return formatted;
  };

  // Add user message and get formatted prompt
  std::string prompt = chat_add_and_format("user", user_input);

  NN_DBG_PRINTF("Processing prompt for session %d: %s", exec_ctx,
                prompt.c_str());

  // Clear KV cache for session isolation (as per user's requirement)
  llama_memory_clear(llama_get_memory(chat_ctx->server_ctx.ctx), true);

  // Tokenize the complete conversation history
  common_chat_templates_inputs inputs;
  inputs.messages = chat_msgs;
  inputs.add_generation_prompt = true;

  // Check if chat templates are available before using them
  if (!chat_ctx->server_ctx.chat_templates.get()) {
    NN_ERR_PRINTF("Chat templates not initialized for prompt generation");
    return "Error: Chat templates not available";
  }

  std::string full_prompt =
      common_chat_templates_apply(chat_ctx->server_ctx.chat_templates.get(), inputs)
          .prompt;

  // Tokenize
  std::vector<llama_token> tokens =
      common_tokenize(chat_ctx->server_ctx.ctx, full_prompt, true, true);

  // Generate response (simplified version of main.cpp's loop)
  std::string response;

  // Process input tokens
  llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

  if (llama_decode(chat_ctx->server_ctx.ctx, batch))
  {
    NN_ERR_PRINTF("Failed to decode input tokens");
    return "Error: Failed to process input";
  }

  // Generate tokens one by one
  for (int i = 0; i < chat_ctx->server_ctx.params_base.n_predict; ++i)
  {
    // Verify that slots[0] and its sampler are valid
    if (chat_ctx->server_ctx.slots.empty() || chat_ctx->server_ctx.slots[0].smpl == nullptr) {
      NN_ERR_PRINTF("Invalid slot or sampler state");
      return "Error: Invalid sampler state";
    }

    llama_token new_token =
        common_sampler_sample(chat_ctx->server_ctx.slots[0].smpl, chat_ctx->server_ctx.ctx, -1);

    if (llama_vocab_is_eog(chat_ctx->server_ctx.vocab, new_token))
    {
      break;
    }

    // Convert token to text
    char buf[256];
    int n = llama_token_to_piece(chat_ctx->server_ctx.vocab, new_token, buf, sizeof(buf),
                                 0, true);
    if (n > 0)
    {
      response.append(buf, n);
    }

    // Prepare next batch
    batch = llama_batch_get_one(&new_token, 1);
    if (llama_decode(chat_ctx->server_ctx.ctx, batch))
    {
      NN_ERR_PRINTF("Failed to decode generated token");
      break;
    }
  }

  // Add assistant response to chat history
  chat_add_and_format("assistant", response);

  return response;
}

// Enhanced helper function to run inference with runtime parameter support
static std::string run_inference_for_session_with_params(LlamaChatContext *chat_ctx,
                                                        graph_execution_context exec_ctx,
                                                        const std::string &user_input,
                                                        const wasi_nn_runtime_params *runtime_params = nullptr)
{
  // Find session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
  {
    return "Error: Invalid session";
  }

  SessionInfo &session_info = session_it->second;
  auto &chat_msgs = session_info.chat_history;

  // Update last activity
  session_info.last_activity = std::chrono::steady_clock::now();

  // Determine max_tokens for this generation
  int max_tokens = chat_ctx->server_ctx.params_base.n_predict;
  if (runtime_params && runtime_params->max_tokens > 0) {
    max_tokens = runtime_params->max_tokens;
    WASI_NN_LOG_DEBUG(chat_ctx, "Using runtime max_tokens: %d", max_tokens);
  }

  // Store original sampling parameters for restoration
  common_params original_params = chat_ctx->server_ctx.params_base;

  // Chat formatting function (from main.cpp)
  auto chat_add_and_format = [&](const std::string &role,
                                 const std::string &content)
  {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;

    // Check if chat templates are available
    if (!chat_ctx->server_ctx.chat_templates.get()) {
      NN_ERR_PRINTF("Chat templates not initialized");
      return std::string("Error: Chat templates not available");
    }

    auto formatted = common_chat_format_single(
        chat_ctx->server_ctx.chat_templates.get(), chat_msgs, new_msg, role == "user",
        false // use_jinja
    );

    chat_msgs.push_back(new_msg);
    return formatted;
  };

  // Add user message and get formatted prompt
  std::string prompt = chat_add_and_format("user", user_input);

  WASI_NN_LOG_DEBUG(chat_ctx, "Processing prompt for session %d: %s", exec_ctx, prompt.c_str());

  // Only clear KV cache if this is a new session (empty chat history)
  // This preserves conversation context for ongoing sessions
  if (chat_msgs.size() <= 1) {  // Only user message just added
    WASI_NN_LOG_DEBUG(chat_ctx, "New session detected, clearing KV cache for session %d", exec_ctx);
    llama_memory_clear(llama_get_memory(chat_ctx->server_ctx.ctx), true);
  } else {
    WASI_NN_LOG_DEBUG(chat_ctx, "Continuing session %d with %zu messages, preserving KV cache",
                       exec_ctx, chat_msgs.size());
  }

  // Tokenize the complete conversation history
  common_chat_templates_inputs inputs;
  inputs.messages = chat_msgs;
  inputs.add_generation_prompt = true;

  // Check if chat templates are available before using them
  if (!chat_ctx->server_ctx.chat_templates.get()) {
    NN_ERR_PRINTF("Chat templates not initialized for prompt generation");
    return "Error: Chat templates not available";
  }

  std::string full_prompt =
      common_chat_templates_apply(chat_ctx->server_ctx.chat_templates.get(), inputs)
          .prompt;

  // Tokenize
  std::vector<llama_token> tokens =
      common_tokenize(chat_ctx->server_ctx.ctx, full_prompt, true, true);

  // Apply runtime parameters to sampler if provided
  if (runtime_params && !chat_ctx->server_ctx.slots.empty() && chat_ctx->server_ctx.slots[0].smpl) {
    apply_runtime_params_to_sampling(chat_ctx->server_ctx.slots[0].smpl, *runtime_params,
                                    chat_ctx->server_ctx.model, chat_ctx);
  }

  // Apply stop sequences if provided
  std::vector<std::string> original_antiprompt;
  if (runtime_params && runtime_params->stop_sequences_set) {
    // Temporarily replace antiprompt with runtime stop sequences
    original_antiprompt = chat_ctx->server_ctx.params_base.antiprompt;
    chat_ctx->server_ctx.params_base.antiprompt = runtime_params->stop_sequences;
    WASI_NN_LOG_DEBUG(chat_ctx, "Applied %zu runtime stop sequences", runtime_params->stop_sequences.size());
  }

  // Generate response (simplified version of main.cpp's loop)
  std::string response;

  // Process input tokens in batches
  const int n_batch = chat_ctx->server_ctx.params_base.n_batch;  // Or wherever n_batch is stored
  size_t n_tokens_processed = 0;
  while (n_tokens_processed < tokens.size()) {
      size_t n_tokens_batch = std::min(static_cast<size_t>(n_batch), tokens.size() - n_tokens_processed);

      llama_batch batch = llama_batch_get_one(tokens.data() + n_tokens_processed, n_tokens_batch);

      if (llama_decode(chat_ctx->server_ctx.ctx, batch)) {
          NN_ERR_PRINTF("Failed to decode input tokens batch");
          return "Error: Failed to process input";
      }

      n_tokens_processed += n_tokens_batch;
  }

  // Generate tokens one by one with runtime parameter support
  for (int i = 0; i < max_tokens; ++i)
  {
    // Verify that slots[0] and its sampler are valid
    if (chat_ctx->server_ctx.slots.empty() || chat_ctx->server_ctx.slots[0].smpl == nullptr) {
      NN_ERR_PRINTF("Invalid slot or sampler state");
      return "Error: Invalid sampler state";
    }

    llama_token new_token =
        common_sampler_sample(chat_ctx->server_ctx.slots[0].smpl, chat_ctx->server_ctx.ctx, -1);

    // Check for EOS token (with runtime ignore_eos support)
    bool should_stop_eos = llama_vocab_is_eog(chat_ctx->server_ctx.vocab, new_token);
    if (runtime_params && runtime_params->ignore_eos_set) {
      should_stop_eos = should_stop_eos && !runtime_params->ignore_eos;
    }

    if (should_stop_eos) {
    //   WASI_NN_LOG_DEBUG(chat_ctx, "Generation stopped at EOS token");
      break;
    }

    // Convert token to text
    char buf[256];
    int n = llama_token_to_piece(chat_ctx->server_ctx.vocab, new_token, buf, sizeof(buf),
                                 0, true);
    if (n > 0)
    {
      response.append(buf, n);
    }

    // Check for stop sequences if provided
    if (runtime_params && runtime_params->stop_sequences_set) {
      for (const auto& stop_seq : runtime_params->stop_sequences) {
        if (response.find(stop_seq) != std::string::npos) {
          WASI_NN_LOG_DEBUG(chat_ctx, "Generation stopped by stop sequence: %s", stop_seq.c_str());
          // Remove the stop sequence from the response
          size_t pos = response.find(stop_seq);
          if (pos != std::string::npos) {
            response = response.substr(0, pos);
          }
          goto generation_complete;
        }
      }
    }

    // Prepare next batch
    llama_batch batch = llama_batch_get_one(&new_token, 1);
    if (llama_decode(chat_ctx->server_ctx.ctx, batch))
    {
      NN_ERR_PRINTF("Failed to decode generated token");
      break;
    }
  }

generation_complete:
  // Restore original antiprompt if we modified it
  if (runtime_params && runtime_params->stop_sequences_set) {
    chat_ctx->server_ctx.params_base.antiprompt = original_antiprompt;
  }

  // Add assistant response to chat history
  chat_add_and_format("assistant", response);

  return response;
}

__attribute__((visibility("default"))) wasi_nn_error
run_inference(void *ctx, graph_execution_context exec_ctx, uint32_t index,
              tensor *input_tensor, tensor_data output_tensor,
              uint32_t *output_tensor_size,
              const char *runtime_config, uint32_t config_len)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !chat_ctx->server_ctx.ctx)
  {
    return invalid_argument;
  }

  // --- START FIX ---
  // 1. Capture the provided capacity of the output buffer before it's used.
  const uint32_t output_buffer_capacity = (output_tensor_size != nullptr) ? *output_tensor_size : 0;
  // --- END FIX ---

  char *prompt_text = (char *)input_tensor->data;
  if (!prompt_text)
  {
    return invalid_argument;
  }

  try
  {
    // Parse runtime parameters if provided
    wasi_nn_runtime_params runtime_params;
    bool params_valid = true;

    if (runtime_config && config_len > 0) {
      params_valid = parse_runtime_params(runtime_config, config_len, runtime_params, chat_ctx);
      if (!params_valid) {
        WASI_NN_LOG_ERROR(chat_ctx, "Failed to parse runtime configuration, using defaults");
        // Continue with default parameters rather than failing
      } else {
        WASI_NN_LOG_INFO(chat_ctx, "Runtime configuration applied successfully");
      }
    }

    // Run inference with enhanced function
    std::string response = run_inference_for_session_with_params(
        chat_ctx, exec_ctx, prompt_text,
        (params_valid && (runtime_config && config_len > 0)) ? &runtime_params : nullptr);

    // --- START FIX ---
    // 2. Use the captured buffer capacity for a safe copy. This prevents writing past
    //    the end of the buffer if the response is larger than the allocated space.
    copy_string_to_tensor_data(output_tensor, output_buffer_capacity, response);

    // 3. Report back the actual number of characters in the response string.
    //    If the response was truncated, the caller can detect it by comparing
    //    this value with the original buffer capacity.
    *output_tensor_size = response.length();
    // --- END FIX ---

    WASI_NN_LOG_DEBUG(chat_ctx, "Generated response: %s", response.c_str());
    return success;
  }
  catch (const std::exception &e)
  {
    WASI_NN_LOG_ERROR(chat_ctx, "Inference failed: %s", e.what());
    return runtime_error;
  }
}

// Placeholder implementations for compatibility
__attribute__((visibility("default"))) wasi_nn_error
load(void *ctx, graph_builder_array *builder, graph_encoding encoding,
     execution_target target, graph *g)
{
  return unsupported_operation;
}

__attribute__((visibility("default"))) wasi_nn_error
load_by_name(void *ctx, const char *filename, uint32_t filename_len, graph *g)
{
  // Use default config
  return load_by_name_with_config(ctx, filename, filename_len, nullptr, 0, g);
}

__attribute__((visibility("default"))) wasi_nn_error
set_input(void *ctx, graph_execution_context exec_ctx, uint32_t index,
          tensor *wasi_nn_tensor)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx || !wasi_nn_tensor)
    return invalid_argument;

  // Find the session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
    return invalid_argument;

  // Get the input prompt from tensor data
  const char *prompt_str = (const char *)wasi_nn_tensor->data;
  if (!prompt_str)
    return invalid_argument;

  // Calculate tensor size from dimensions
  uint32_t tensor_size = 1;
  if (wasi_nn_tensor->dimensions && wasi_nn_tensor->dimensions->buf) {
    for (uint32_t i = 0; i < wasi_nn_tensor->dimensions->size; ++i) {
      tensor_size *= wasi_nn_tensor->dimensions->buf[i];
    }
  }

  // For text data, we assume it's null-terminated or use a reasonable max length
  size_t prompt_len = strnlen(prompt_str, tensor_size);
  std::string prompt(prompt_str, prompt_len);

  // Store the prompt for later processing in compute()
  // For now, we'll store it in the session (could be optimized)
  session_it->second.session_id = prompt; // Temporary storage

  NN_INFO_PRINTF("Input set for execution context %d: %.100s%s",
                 exec_ctx, prompt.c_str(),
                 prompt.length() > 100 ? "..." : "");

  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
compute(void *ctx, graph_execution_context exec_ctx)
{
  LlamaChatContext *chat_ctx = (LlamaChatContext *)ctx;
  if (!chat_ctx)
    return invalid_argument;

  // Phase 4.3: Automatic memory optimization before processing
  wasi_nn_error opt_result = auto_optimize_memory(chat_ctx, exec_ctx);
  if (opt_result != success) {
    NN_WARN_PRINTF("Memory optimization warning for session %u: %d", exec_ctx, opt_result);
    // Continue with inference even if optimization has issues
  }

  // Find the session
  auto session_it = chat_ctx->sessions.find(exec_ctx);
  if (session_it == chat_ctx->sessions.end())
    return invalid_argument;

  // For simplified version, process immediately without concurrency limits
  NN_INFO_PRINTF("Processing compute request immediately for execution context %d", exec_ctx);

  // Update last activity time
  session_it->second.last_activity = std::chrono::steady_clock::now();

  // Phase 4.3: Auto context shift if needed (context window approaching limit)
  auto_perform_context_shift_session(chat_ctx, exec_ctx);

  // For Phase 4.2, we're mainly implementing the queuing mechanism
  // The actual inference processing remains the same as before
  // This would typically involve:
  // 1. Creating server tasks
  // 2. Processing through the server context
  // 3. Managing the llama context and sampling

  return success;
}

__attribute__((visibility("default"))) wasi_nn_error
get_output(void *ctx, graph_execution_context exec_ctx, uint32_t index,
           tensor_data output_tensor, uint32_t *output_tensor_size)
{
  return success;
}

// Phase 4.3: Internal Memory Management Functions
// ===============================================
// These functions are automatically called during inference for optimization

static wasi_nn_error
auto_clear_kv_cache_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }

  NN_DBG_PRINTF("Auto-clearing KV cache for session %u", exec_ctx);

  wasi_nn_error result = clear_kv_cache(chat_ctx, exec_ctx);
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-clear KV cache for session %u: %d", exec_ctx, result);
    return result;
  }

  return success;
}

static wasi_nn_error
auto_clear_all_kv_cache(LlamaChatContext *chat_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }

  NN_DBG_PRINTF("Auto-clearing all KV cache");

  wasi_nn_error result = clear_kv_cache(chat_ctx, 0); // session_id = 0 means all sessions
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-clear all KV cache: %d", result);
    return result;
  }

  return success;
}

static wasi_nn_error
auto_perform_context_shift_session(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }

  if (!chat_ctx->context_shifting_enabled) {
    NN_DBG_PRINTF("Context shifting is disabled for session %u", exec_ctx);
    return success; // Not an error, just disabled
  }

  NN_DBG_PRINTF("Auto-performing context shift for session %u", exec_ctx);

  wasi_nn_error result = perform_context_shift(chat_ctx, exec_ctx);
  if (result != success) {
    NN_WARN_PRINTF("Failed to auto-perform context shift for session %u: %d", exec_ctx, result);
    return result;
  }

  return success;
}

static wasi_nn_error
auto_optimize_memory(LlamaChatContext *chat_ctx, graph_execution_context exec_ctx)
{
  if (!chat_ctx) {
    NN_ERR_PRINTF("Invalid context");
    return invalid_argument;
  }

  NN_DBG_PRINTF("Auto-optimizing memory for session %u", exec_ctx);

  // Check for memory pressure and handle it
  if (check_memory_pressure(chat_ctx)) {
    NN_INFO_PRINTF("Memory pressure detected, performing automatic cleanup");
    wasi_nn_error result = handle_memory_pressure(chat_ctx);
    if (result != success) {
      NN_WARN_PRINTF("Failed to handle memory pressure: %d", result);
      // Don't fail the inference, just log warning
    }
  }

  // Optimize token cache (non-critical)
  wasi_nn_error result = optimize_token_cache(chat_ctx, exec_ctx);
  if (result != success) {
    NN_DBG_PRINTF("Token cache optimization skipped: %d", result);
    // This is not critical for inference
  }

  return success;
}
