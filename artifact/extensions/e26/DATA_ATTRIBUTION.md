# Public data attribution

Friendster workload bytes in `packet/workloads/friendster_pairs.npy` derive from
Per Fuchs, *Com-Friendster Dataset Densely Relabelled* (2021),
[Zenodo 5146230](https://doi.org/10.5281/zenodo.5146230), licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The study authors
generated the fixed query sequence; this extension copies those original bytes.
The original network is J. Yang and J. Leskovec, *Defining and Evaluating Network
Communities based on Ground-truth*, ICDM 2012;
[SNAP source](https://snap.stanford.edu/data/com-Friendster.html).

Citation2 workload bytes in `packet/workloads/citation2_pairs.npy` derive from
the public [OGB ogbl-citation2 release](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2),
licensed [ODC-BY](https://opendatacommons.org/licenses/by/1-0/). The study authors
generated the fixed strict-clean query sequence; this extension copies those
original bytes. OGB's MIT software license is distinct from its dataset license.

Full graphs, features, native representations, model checkpoints and private
industrial data are absent. Follow the base artifact's `DATA_ACCESS.md` and
`PUBLIC_COMMANDS.txt` for public reconstruction and exact numeric identities.
Retain this attribution when redistributing the identical small workload files.
