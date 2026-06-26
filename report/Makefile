MAIN = main
LATEX = xelatex
BIBER = biber
LATEX_FLAGS = -halt-on-error -file-line-error

.PHONY: all clean force

all:
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex && $(BIBER) $(MAIN) && $(LATEX) $(LATEX_FLAGS) $(MAIN).tex && $(LATEX) $(LATEX_FLAGS) $(MAIN).tex

force: clean all

clean:
	-del /S *.pdf *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc *.lof *.lot *.xdv *.tdo *.acn *.acr *.alg 2>nul
