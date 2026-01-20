
export OPENPI_DATA_HOME=/home/dataset-local/cache
uv run scripts/compute_norm_stats.py --config-name pi05_franka
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_franka --exp-name=plug_peg