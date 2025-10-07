default: calculus

calculus: calculus.md
	pandoc -f markdown --filter pandoc-sidenote calculus.md -o calculus.html --css tufte.css --lua-filter=../section-filter.lua --mathjax --standalone
