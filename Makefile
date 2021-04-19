EXTENSION = c
CC = gcc
SRCDIR = src
OBJDIR = obj
DEPDIR = dep
INCLUDEDIR = include

CPPFLAGS = $(addprefix -I,$(INCLUDEDIR)) 
LDFLAGS = -lm

CFLAGS += -O3 -Wall -D_GNU_SOURCE
 
EXE = lr_mbsgd
CXX_SOURCES = $(foreach dir,$(SRCDIR), $(wildcard $(dir)/*.$(EXTENSION)))
CXX_OBJECTS = $(patsubst  %.$(EXTENSION), $(OBJDIR)/%.o, $(notdir $(CXX_SOURCES)))
DEP_FILES  = $(patsubst  %.$(EXTENSION), $(DEPDIR)/%.d, $(notdir $(CXX_SOURCES)))

CUDA_EXE = lrgpu_mbsgd
CUDA_TARGET = libculrmbsgd.a
CUDA_EXTENSION = cu
CUDA_SRCDIR = cuda
CUDA_OBJDIR = $(OBJDIR)/cuda
CUDA_SOURCES = $(foreach dir,$(CUDA_SRCDIR), $(wildcard $(dir)/*.$(CUDA_EXTENSION)))
CUDA_OBJECTS = $(patsubst  %.$(CUDA_EXTENSION), $(CUDA_OBJDIR)/%.o, $(notdir $(CUDA_SOURCES)))

ifeq "$(MAKECMDGOALS)" "cuda"
CFLAGS += -D_CUDA

AR = ar
 
CCPLUS = g++
NVCC = nvcc
CUDA_FLAGS = --compiler-options -fPIC -shared

#CXX_SOURCES := src/main.c src/lr.c
CXX_OBJECTS := obj/cuda/main.o obj/cuda/lr.o 

#$(CXX_OBJECTS): $(CXX_SOURCES)
#	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/cuda/main.o: src/main.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

obj/cuda/lr.o: src/lr.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

$(CUDA_TARGET): $(CUDA_OBJECTS)
	$(AR) rcs $@ $^

$(CUDA_OBJDIR)/%.o: $(CUDA_SRCDIR)/%.$(CUDA_EXTENSION)
	$(NVCC) -D_CUDA $(CUDA_FLAGS) $(CPPFLAGS) -c $^ -o $@

$(CUDA_EXE): $(CXX_OBJECTS) $(CUDA_TARGET)
	$(CCPLUS) $(CFLAGS) $(CFLAGS) $(LDFLAGS) $(CXX_OBJECTS) -L. -l:$(CUDA_TARGET) -L/usr/local/cuda/lib64 -lcudart -o $@
else
LDFLAGS += -lpthread

$(OBJDIR)/%.o: $(SRCDIR)/%.$(EXTENSION)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
 
$(DEPDIR)/%.d: $(SRCDIR)/%.$(EXTENSION)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MM $< | sed -e 1's,^,$(OBJDIR)/,' > $@

$(EXE): $(CXX_OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(CXX_OBJECTS) -o $@
	
endif


ifneq "$(MAKECMDGOALS)" "clean"
-include $(DEP_FILES)
endif

.PHONY: cuda
cuda: $(CUDA_EXE)

.PHONY: clean
clean:
	-rm -f $(CXX_OBJECTS) $(DEP_FILES) $(EXE) $(CUDA_OBJECTS) $(CUDA_TARGET) $(CUDA_EXE)
