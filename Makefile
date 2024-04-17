.PHONY: all extract balance analysis

all: template extract build balance analysis

template:
	python main/create_template.py
	python main/create_treatments.py

extract:
    python main/run_extract.py

build:
    python main/build_dataset.py

balance:
    python main/run_balancing.py

analysis:
    python main/run_analysis.py

clean:
    rm -r outputs/*