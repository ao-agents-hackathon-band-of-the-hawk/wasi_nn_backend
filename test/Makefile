# Makefile for WASI-NN Backend Test Suite (Modular)
# Usage: make test && ./main

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g -O2
LDFLAGS = -ldl -lpthread

# Modular source files
COMMON_SOURCES = test_common.c
TEST_SOURCES = test_basic.c test_inference.c test_session.c test_logging.c test_model.c test_stopping.c test_error.c test_lora.c
MAIN_SOURCE = main.c
ALL_SOURCES = $(COMMON_SOURCES) $(TEST_SOURCES) $(MAIN_SOURCE)

# Targets
TARGET = main
TARGET_ORIGINAL = main_original

# Default target uses modular architecture
$(TARGET): $(ALL_SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(ALL_SOURCES) $(LDFLAGS)

# Original monolithic version (backup)
$(TARGET_ORIGINAL): main.c
	$(CC) $(CFLAGS) -o $(TARGET_ORIGINAL) main.c $(LDFLAGS)

# Test target
test: $(TARGET)
	@echo "✅ Modular test executable built successfully"
	@echo "Run with: ./$(TARGET)"

# Test original version
test-original: $(TARGET_ORIGINAL)
	@echo "✅ Original test executable built successfully"
	@echo "Run with: ./$(TARGET_ORIGINAL)"

# Test both versions
test-all: $(TARGET) $(TARGET_ORIGINAL)
	@echo "✅ Both test executables built successfully"
	@echo "Run modular version: ./$(TARGET)"
	@echo "Run original version: ./$(TARGET_ORIGINAL)"

# Clean target
clean:
	rm -f $(TARGET) $(TARGET_ORIGINAL) *.o
	@echo "✅ Cleaned all test executables and object files"

# Install target (ensure backend library exists)
install:
	@echo "Checking backend library..."
	@if [ ! -f "../build/libwasi_nn_backend.so" ]; then \
		echo "❌ Backend library not found. Please run 'make -C .. -j16' first"; \
		exit 1; \
	else \
		echo "✅ Backend library found"; \
	fi

# Full build and test
all: install test
	@echo "✅ Ready to run modular tests with: ./$(TARGET)"

# Build all versions
all-versions: install test-all
	@echo "✅ Ready to run tests:"
	@echo "  Modular: ./$(TARGET)"
	@echo "  Original: ./$(TARGET_ORIGINAL)"

.PHONY: test test-original test-all clean install all all-versions
