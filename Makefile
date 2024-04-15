.PHONY: all extract balance analysis

all: extract balance analysis

template:
	ython main/create_template.py
	python main/create_treatments.py

extract:
    python src/run_extract.py

balance:
    python src/run_balancing.py

analysis:
    python src/run_analysis.py