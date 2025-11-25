.PHONY: default clean

# Source files
SOURCES := calculus/calculus.md linear-algebra/linear-algebra.md probability/probability.md

# Outputs go in top level: strip directories
OUTPUTS := $(notdir $(SOURCES:.md=.html))

CSS := tufte.css
LUA := section-filter.lua

# Default builds all outputs
default: $(OUTPUTS)

# Rule: src/dir/file.md -> file.html (in top-level)
%.html: */%.md $(CSS) $(LUA)
	# must start with a TAB
	pandoc -f markdown --filter pandoc-sidenote $< \
	    -o $@ \
	    --css $(CSS) \
	    --lua-filter=$(LUA) \
	    --mathjax \
	    --standalone

clean:
	rm -f $(OUTPUTS)

