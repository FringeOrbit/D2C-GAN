# Paper results

These files are derived result summaries and figure source files used in the
manuscript. They do not contain the raw well-log datasets.

The result groups are:

- `figures/`: final manuscript figures and compact figure data;
- `metrics/`: internal comparisons, ablations and protocol summaries;
- `external_validation/`: independent-basin summary files and well-level
  aggregate metrics;
- `logs/`: concise experiment records associated with the reported analyses.

The principal manuscript mappings are:

| Manuscript item | Repository location |
| --- | --- |
| Figs. 1--7 | `figures/figure_01_*` through `figures/figure_07_*` |
| Covariance and PSD diagnostics | `figures/figure_05_*`, `figures/figure_06_*`, `figures/figure_07_*` |
| External Espírito Santo evaluation | `external_validation/` |
| Well-level bootstrap summary | `metrics/external_well_level_bootstrap.*` |
| Figure-generation and diagnostic records | `logs/` |

The comparison tables in the manuscript report baseline results, but baseline
implementations and raw datasets are intentionally not distributed here.
