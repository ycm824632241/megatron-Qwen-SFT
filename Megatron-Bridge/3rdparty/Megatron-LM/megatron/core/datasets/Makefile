CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
PYTHON ?= python3
CPPFLAGS += $(shell $(PYTHON) -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
