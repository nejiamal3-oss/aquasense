PYTHON ?= python

.PHONY: all simulate perturb fuse benchmark clean

all: simulate perturb fuse benchmark

simulate:
	$(PYTHON) aquasense/run_simulation.py \
	  --inp network/water_network.inp \
	  --msx network/water_quality.msx \
	  --out simulated_data.csv

perturb:
	$(PYTHON) -m aquasense.perturb --seed 42 \
	  --in Real_data.csv --out attacked_data.csv

fuse:
	$(PYTHON) aquasense/fuse.py \
	  --real Real_data.csv \
	  --sim simulated_data.csv \
	  --out fused_dataset.csv

benchmark:
	$(PYTHON) benchmark/run_t1.py
	$(PYTHON) benchmark/run_t3.py
	$(PYTHON) benchmark/run_t4.py

clean:
	rm -f simulated_data.csv attacked_data.csv fused_dataset.csv
