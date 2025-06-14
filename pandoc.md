# Pandoc tips

To render html page from markdown with mathjax
(standalone was also needed : <https://stackoverflow.com/questions/37533412/md-with-latex-to-html-with-mathjax-with-pandoc> )

```
pandoc --mathjax --standalone -f markdown --filter pandoc-sidenote week1.md -o week1.html
```
