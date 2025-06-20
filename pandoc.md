# Pandoc tips

Filters I'm using
<https://github.com/jez/pandoc-sidenote>

`section-filter.lua` - A filter I created to automatically add `<article>` and `<section>` tags to the various sections marked off by `<body>` and `<h2>`
To render html page from markdown with mathjax
(standalone was also needed : <https://stackoverflow.com/questions/37533412/md-with-latex-to-html-with-mathjax-with-pandoc> )

Command I run:

```
pandoc --mathjax --standalone -f markdown --filter pandoc-sidenote week1.md -o week1.html
```
