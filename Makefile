EXTENSION = c
CC = gcc
EXE = lr_mbsgd
SRCDIR = src
OBJDIR = obj
DEPDIR = dep
INCLUDEDIR = include

#预处理选项（要包含的.h文件的路径）
CPPFLAGS += -I $(INCLUDEDIR) 
#链接选项
LDFLAGS += -lm -lpthread
#编译器的选项
CFLAGS += -O3 -Wall -D_GNU_SOURCE -D_CUDA
 
#后面的内容都不需要修改
CXX_SOURCES = $(foreach dir,$(SRCDIR), $(wildcard $(dir)/*.$(EXTENSION)))
CXX_OBJECTS = $(patsubst  %.$(EXTENSION), $(OBJDIR)/%.o, $(notdir $(CXX_SOURCES)))
DEP_FILES  = $(patsubst  %.$(EXTENSION), $(DEPDIR)/%.d, $(notdir $(CXX_SOURCES)))
 
#CUDA_EXTENSION = cu
#NVCC = nvcc
#CUDA_SRCDIR = cuda
#CUDA_SOURCES = $(foreach dir,$(CUDA_SRCDIR), $(wildcard $(dir)/*.$(CUDA_EXTENSION)))
#CUDA_OBJECTS = $(patsubst  %.$(CUDA_EXTENSION), $(OBJDIR)/%.o, $(notdir $(CUDA_SOURCES)))
#CUDA_DEP_FILES  = $(patsubst  %.$(CUDA_EXTENSION), $(DEPDIR)/%.d, $(notdir $(CUDA_SOURCES)))

$(EXE): $(CXX_OBJECTS) 
	$(CC) $(CFLAGS) $(LDFLAGS) $(CXX_OBJECTS) -o $(EXE)
 
$(OBJDIR)/%.o: $(SRCDIR)/%.$(EXTENSION)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
 
$(DEPDIR)/%.d: $(SRCDIR)/%.$(EXTENSION)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MM $< | sed -e 1's,^,$(OBJDIR)/,' > $@
 
ifneq "$(MAKECMDGOALS)" "clean"
-include $(DEP_FILES)
endif
 
.PHONY: clean
clean:
	-rm -f $(CXX_OBJECTS) $(DEP_FILES) $(EXE)
