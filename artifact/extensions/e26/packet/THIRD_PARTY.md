# Third-party provenance inventory

The authors' own code is MIT under LICENSE, approved by the user on 2026-11-25.
The following upstream notices retain their original copyright holders and
license scopes. Historical version/derivation checks remain explicit where
the exact legacy source ancestry has not been independently reconstructed.

| Component | Version / source | License / treatment |
|---|---|---|
| pytorch_sparse | commit 7d22892fb47626d2c5b0171769d44bdc8edd606c; torch-sparse0.6.18 | MIT, exact upstream copyright/permission text in third_party/pytorch_sparse_LICENSE; receipt gives URL and SHA256 |
| OGB | 1.3.6 | MIT software; install as dependency, not vendored here; Citation2 dataset ODC-BY |
| PyTorch | 2.5.0 | BSD-style dependency; install official binary distribution with its notices |
| PyG | 2.7.0 | MIT dependency; install official distribution with its notices |
| NumPy, SciPy, pandas | pinned in requirements | BSD-style dependencies; distributions retain their notices |
| Numba | 0.58.0 | BSD dependency; distribution retains its notices |
| SEAL/DGCNN-derived model and labeling code | Facebook copyright headers retained; located SEAL_OGB commit ea01ef509df1a90bc246e4e9828e192eeec4289c | Exact MIT text in third_party/SEAL_OGB_LICENSE; historical derivation/version audit still pending |
| VK-GNN serialization-format lineage / related prior implementation | makut/VK-GNN commit 2a06545429210962c94b84722376154d8a8c4602, released May 2026 | Apache-2.0; exact text in third_party/VK_GNN_LICENSE. Shared binary layout is acknowledged; no direct source-copy or translation direction is asserted. See metadata/PRIOR_WORK_PROVENANCE.json. |
| PENCIL bridge and corrected native control | Exact source pins in metadata/SYSTEMS_FROZEN_INPUTS.json | Authors' bridge/adapters MIT; upstream PENCIL Apache-2.0 and torch-sparse MIT notices retained. Source integration does not prove full-scale result reproduction. |

Verified pytorch_sparse LICENSE SHA256:
`7cbd1d9075b9cb3daa771f52d10e1a6b9c4577851f055d002449932fee4d7486`.
Its exact copyright line is preserved verbatim. Dataset attributions are in
DATA_ACCESS.md.

VK-GNN already has int32 neighbors, int64 offsets, raw N offsets without a
terminal sentinel, the same edge_starts/edge_ends/timestamps filenames and an
int64 element-count header for the int32 arrays. Its inspected Java readers
allocate off-heap memory and read from FileChannel; this evidence does not
establish file-backed mmap there. LPP's Python offset-array documentation
explicitly describes Java compatibility. Its recorded Git history already
contains the Python mmap loader in February 2026, before the public VK-GNN
release. Exact common ancestry and translation direction are not established
by those observations.

The shared CSR format and pre-existing Python loader must not be presented as
new inventions of this study. The new study's fixed-set induction, execution
controls and sampling analysis require their own evidence. MIT remains the
approved license for the authors' own code; it does not remove any applicable
Apache-2.0 obligations for inherited material. No VK-GNN Java/C++ source is
included in this staged artifact. The exact prior license is retained as
provenance, and no upstream root NOTICE file was found at the pinned commit.

## Candidate systems additions

PENCIL pinned Apache-2.0 sources and exact license are included under
work/vldb_pencil_bridge/upstream_pencil. Native MIT source notices are retained
under work/vldb_corrected_native/third_party. See PACKAGING_NOTES.md for precise
own-code/upstream/BERT-config scopes. Integration of sources does not attest
completion or correctness of the pending full-scale systems results.
