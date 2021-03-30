EXTENSION = c
CC = gcc
SRCDIR = src
OBJDIR = obj
DEPDIR = dep
INCLUDEDIR = include

CPPFLAGS = $(addprefix -I,$(INCLUDEDIR)) 
LDFLAGS = -lm -lpthread

CFLAGS += -O3 -Wall -D_GNU_SOURCE
 
CXX_SOURCES = $(foreach dir,$(SRCDIR), $(wildcard $(dir)/*.$(EXTENSION)))
CXX_OBJECTS = $(patsubst  %.$(EXTENSION), $(OBJDIR)/%.o, $(notdir $(CXX_SOURCES)))
DEP_FILES  = $(patsubst  %.$(EXTENSION), $(DEPDIR)/%.d, $(notdir $(CXX_SOURCES)))


$(OBJDIR)/%.o: $(SRCDIR)/%.$(EXTENSION)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
 
$(DEPDIR)/%.d: $(SRCDIR)/%.$(EXTENSION)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MM $< | sed -e 1's,^,$(OBJDIR)/,' > $@

ifeq "$(MAKECMDGOALS)" "cuda"
CUDA_EXE = lrgpu_mbsgd
CFLAGS += -D_CUDA

AR = ar
 
CUDA_EXTENSION = cu
NVCC = nvcc
CUDA_TARGET = libculrmbsgd.a
CUDA_SRCDIR = cuda
CUDA_OBJDIR = $(OBJDIR)/cuda
CUDA_SOURCES = $(foreach dir,$(CUDA_SRCDIR), $(wildcard $(dir)/*.$(CUDA_EXTENSION)))
CUDA_OBJECTS = $(patsubst  %.$(CUDA_EXTENSION), $(CUDA_OBJDIR)/%.o, $(notdir $(CUDA_SOURCES)))
CUDA_FLAGS = --compiler-options -fPIC -shared

$(CUDA_TARGET): $(CUDA_OBJECTS)
	$(AR) rcs $@ $^

$(CUDA_OBJDIR)/%.o: $(CUDA_SRCDIR)/%.$(CUDA_EXTENSION)
	$(NVCC) $(CUDA_FLAGS) $(CPPFLAGS) -c $< -o $@

$(CUDA_EXE): $(CXX_OBJECTS) $(CUDA_TARGET)
	$(CC) $(CFLAGS) $(LDFLAGS) $(CXX_OBJECTS) -L. -l:$(CUDA_TARGET) -L/usr/local/cuda/lib64 -lcudart -o $@

else
EXE = lr_mbsgd
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
	-rm -f $(CXX_OBJECTS) $(DEP_FILES) $(CUDA_OBJECTS) $(EXE) $(CUDA_TARGET)
