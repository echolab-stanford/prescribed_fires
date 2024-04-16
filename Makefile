.PHONY: all extract balance analysis

all: extract balance analysis

template:
	ython main/create_template.py
	python main/create_treatments.py

extract:
    python main/run_extract.py

balance:
    python main/run_balancing.py

analysis:
    python main/run_analysis.py