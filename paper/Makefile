all: paper.pdf

paper.pdf: paper.md paper.bib paper.xml
	pandoc -o paper.pdf -V geometry:margin=1in --filter pandoc-citeproc --bibliography paper.bib paper.md --template latex.template

paper.xml: paper.md paper.bib
	pandoc -s -f markdown paper.md -o paper.xml --template xml.template

clean:
	rm paper.xml
	rm paper.pdf


.PHONY: clean
