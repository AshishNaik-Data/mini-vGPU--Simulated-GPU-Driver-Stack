# mini vGPU - builds with clang or gcc on macOS and Linux.
#
#   make           build everything into build/
#   make test      run the driver tests
#   make asan      tests under AddressSanitizer + UBSan (memory bugs)
#   make tsan      tests under ThreadSanitizer (data races)
#   make bench     CPU vs vGPU edge pipeline, writes build/edges.pgm
#   make sched     compare the five scheduling policies
#   make tlb       TLB hit rate with 4 KiB vs 64 KiB pages
#   make layout    kernel time vs distance between two buffers (finding A)
#   make clean

CC    ?= cc
CXX   ?= c++
BUILD ?= build
SAN   ?=
OPT   ?= -O2 -g

CPPFLAGS += -Iinclude -Isrc/driver -MMD -MP
CFLAGS   += -std=c11 -Wall -Wextra -Wpedantic -Wshadow $(OPT) $(SAN) -pthread
CXXFLAGS += -std=c++17 -Wall -Wextra -Wpedantic -Wshadow $(OPT) $(SAN) -pthread
LDFLAGS  += $(SAN) -pthread

DRIVER_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(wildcard src/driver/*.c))
LIB        := $(BUILD)/libvgpu.a

.PHONY: all test asan tsan bench sched tlb layout clean

all: $(BUILD)/test_driver $(BUILD)/edge_bench $(BUILD)/sched_demo $(BUILD)/tlb_demo $(BUILD)/layout_sweep

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(LIB): $(DRIVER_OBJ)
	ar rcs $@ $^

$(BUILD)/test_driver: $(BUILD)/tests/test_driver.o $(LIB)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/sched_demo: $(BUILD)/tools/sched_demo.o $(LIB)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/tlb_demo: $(BUILD)/tools/tlb_demo.o $(LIB)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/layout_sweep: $(BUILD)/tools/layout_sweep.o $(LIB)
	$(CC) $^ $(LDFLAGS) -o $@

$(BUILD)/edge_bench: $(BUILD)/cpp/edge_bench.o $(LIB)
	$(CXX) $^ $(LDFLAGS) -o $@

test: $(BUILD)/test_driver
	./$(BUILD)/test_driver

asan:
	$(MAKE) BUILD=build-asan OPT="-O1 -g" SAN="-fsanitize=address,undefined -fno-omit-frame-pointer" test

tsan:
	$(MAKE) BUILD=build-tsan OPT="-O1 -g" SAN="-fsanitize=thread" test

bench: $(BUILD)/edge_bench
	./$(BUILD)/edge_bench --width 1920 --height 1080 --iters 10 --out $(BUILD)/edges.pgm

sched: $(BUILD)/sched_demo
	./$(BUILD)/sched_demo

tlb: $(BUILD)/tlb_demo
	./$(BUILD)/tlb_demo

layout: $(BUILD)/layout_sweep
	./$(BUILD)/layout_sweep

clean:
	rm -rf build build-asan build-tsan

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
