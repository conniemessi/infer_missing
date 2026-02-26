# infer_missing

This repository is the official implementation of:

> Wendi Ren, Ke Wan, Junyu Leng, Shuang Li.  
> **Inferring the Invisible: Neuro-Symbolic Rule Discovery for Missing Value Imputation.**  
> ICLR 2026.

## Environment

This codebase is implemented in Python and uses PyTorch. A minimal setup is:

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- psutil

You can create a virtual environment and install the dependencies with:

```bash
pip install torch numpy matplotlib psutil
```

## Quick start

We provide the code for synthetic setting (b) of Figure 3. The main entry point for running experiments on the synthetic rule-based benchmark is:

```bash
python main_rule_b.py \
  --seed 42 \
  --obs_prob 0.3 \
  --n_samples 20000 \
  --epochs_per_block 30 \
  --max_cycles 5
```

Key arguments:

- `--obs_prob`: probability that a hidden predicate is observed,
- `--missing_mechanism`: missing data mechanism (`MCAR`, `MAR`, or `MNAR`),
- `--beta_mode`: temperature schedule for the OR over rules (`constant` or `cycle`),
- `--softmin_temp`: temperature for the differentiable AND/softmin operator.

The script saves:

- learned rule structures and selected predicates (JSON) under `output/results_rule_b/`, and
- training curves and accuracy plots (PDF) under `output/results_rule_b/plots/`.

Run the following command to see all options:

```bash
python main_rule_b.py -h
```

## Citation

If you find this code useful in your research, please cite:

```bibtex
@inproceedings{ren2026inferring,
  title     = {Inferring the Invisible: Neuro-Symbolic Rule Discovery for Missing Value Imputation},
  author    = {Ren, Wendi and Wan, Ke and Leng, Junyu and Li, Shuang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
