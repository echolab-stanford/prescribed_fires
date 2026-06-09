# Figure → notebook map

Mapping of each figure in the paper (`low_severity_fires.pdf`) to the notebook
and cell that generates it. Cell numbers are 0-indexed positions in the `.ipynb`.

## Main figures

| Paper figure | Notebook | Cell(s) |
| --- | --- | --- |
| Fig 1 (a–d) | `plot_data.ipynb` | 9, 10, 11, 12 |
| Fig 1e | `plot_method.ipynb` | 3, 6, 7, 8, 9 |
| Fig 2 | `results_figures_replication.ipynb` | 19 \* |
| Fig 3 | `spillovers.ipynb` | 22 |
| Fig 4 | `simulation_replication.ipynb` | 32 |
| Fig 5a–b | `simulation_replication.ipynb` | 75, 77 |
| Fig 5c | `simulation_replication.ipynb` | 42, 43 |

## Supplementary figures

| Paper figure | Notebook | Cell(s) |
| --- | --- | --- |
| S1 | — (schematic, not notebook-generated) | — |
| S2 | `results_figures_replication.ipynb` | 14, 15 |
| S3 | `results_figures_replication.ipynb` | 22 |
| S4 | `results_figures_replication.ipynb` | 5, 8 |
| S5 | `results_figures_replication.ipynb` | 25 \* |
| S6 | `spillovers.ipynb` | 8, 14 |
| S7 | `spillovers.ipynb` | 24, 25 |
| S8 | `simulation_replication.ipynb` | 62, 66, 69 |
| S9 | `simulation_replication.ipynb` | 20 |
| S10 | `simulation_replication.ipynb` | 16 |
| S11 | `results_figures_replication.ipynb` (a) + `spillovers.ipynb` (b) | 33 |
| S12 | `spillovers.ipynb` | 29, 31 |
| S13 | `simulation_replication.ipynb` | — (section present; not saved to file) |
| S14 | `simulation_replication.ipynb` | 52 |
| S16 | `suppression_plots.ipynb` | 6, 8, 12, 18 |
| S17 | `confidence_frp.ipynb` | 4, 9, 12 |
| S18 | `simulation_replication.ipynb` | 47, 48 |
| S19 | `results_figures_replication.ipynb` | 26, 27 |
| S20 | `simulation_replication.ipynb` | 58, 59 |
| S21 | — (schematic, not notebook-generated) | — |

\* `savefig` is commented out in this cell — the figure renders inline but is not
written to disk on a plain top-to-bottom run.
