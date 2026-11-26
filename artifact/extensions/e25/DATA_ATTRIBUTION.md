# Public data attribution

The only standalone data payload in this source addition is
`source/packet/workloads/friendster_pairs.npy`: 192,128 bytes, SHA256
`340e61cdd1eb8d329d52be164f919480e6a41feb4c620852df0393c2e9a192f0`.
It is the small public Friendster query workload, not the graph, a restricted
industrial split, a checkpoint or a predictive-quality result.

Source: Per Fuchs, *Com-Friendster Dataset Densely Relabelled* (2021),
DOI [10.5281/zenodo.5146230](https://doi.org/10.5281/zenodo.5146230),
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
The study's authors prepared this query workload from that public dense graph.
Original network: J. Yang and J. Leskovec, *Defining and Evaluating Network
Communities based on Ground-truth*, ICDM 2012;
https://snap.stanford.edu/data/com-Friendster.html . Retain this attribution
when reusing the derived workload. Software licenses and dataset attribution
have separate scopes. Full public-graph preparation instructions belong to the
base artifact's `DATA_ACCESS.md`; no full CSR graph is included here.
