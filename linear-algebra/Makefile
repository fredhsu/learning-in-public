default: linear-algebra

linear-algebra: linear-algebra.md
	pandoc -f markdown --filter pandoc-sidenote linear-algebra.md -o linear-algebra.html --css tufte.css --lua-filter=section-filter.lua --mathjax --standalone
