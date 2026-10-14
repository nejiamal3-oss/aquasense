PYTHON ?= python

.PHONY: all simulate perturb fuse benchmark clean

all: simulate perturb fuse benchmark

simulate:
	$(PYTHON) aquasense/run_simulation.py \
	  --inp network/water_network.inp \
	  --msx network/water_quality.msx \
	  --out data/simulated_data.csv

perturb:
	$(PYTHON) -m aquasense.perturb --seed 42 \
	  --in data/Real_data.csv --out data/attacked_data.csv

fuse:
	$(PYTHON) aquasense/fuse.py \
	  --real data/Real_data.csv \
	  --sim data/simulated_data.csv \
	  --out data/fused_dataset.csv

benchmark:
	$(PYTHON) benchmark/run_t1.py
	$(PYTHON) benchmark/run_t3.py
	$(PYTHON) benchmark/run_t4.py

clean:
	rm -f data/simulated_data.csv data/attacked_data.csv data/fused_dataset.csv