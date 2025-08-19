# Compiler for test suite
CXX = g++

# Compiler flags for main project
CXXFLAGS = -Wall -Wextra -Wformat -Wformat-security -ffunction-sections -fdata-sections -Wno-unused-parameter -fvisibility=hidden -DNN_LOG_LEVEL=0

# Linker flags for main project
LDFLAGS = -ldl

all: build

# Build the main WASI-NN backend library
build:
	@echo "Building WASI-NN backend library..."
	@if [ ! -d "build" ]; then mkdir -p build; fi
	cd build && cmake .. && $(MAKE) -j 12
	@echo "✅ WASI-NN backend library built successfully"


# Clean rule - clean both main project and tests
clean:
	rm -rf build
	@echo "✅ Cleaned all build artifacts"

# Help target
help:
	@echo "Available targets:"
	@echo "  build      - Build the WASI-NN backend library"
	@echo "  clean      - Clean all build artifacts"
	@echo "  help       - Show this help message"

.PHONY: all build clean help
