# Notices for this source-only extension

Own code is MIT under [LICENSE](LICENSE). The unchanged scientific packet
retains its original copyright headers, provenance and license files:

- [Packet provenance inventory](packet/THIRD_PARTY.md).
- [SEAL_OGB MIT notice](packet/third_party/SEAL_OGB_LICENSE).
- [pytorch_sparse MIT notice](packet/third_party/pytorch_sparse_LICENSE), also
  retained with the standalone native core under `packet/native_core/third_party/`.
- [VK-GNN Apache-2.0 notice](packet/third_party/VK_GNN_LICENSE) and recorded
  format-lineage metadata. No claim of a new CSR format is made.

The packet inventory also contains artifact-wide historical references to
PENCIL, prior-work metadata, `DATA_ACCESS.md` and `PACKAGING_NOTES.md`. Resolve
those references at the base artifact root. They do not assert that the full
PENCIL codebase or unrelated prior experiments are vendored in this extension.
Installed numerical dependencies retain their own distribution notices.

The two small frozen workload files are derived from public datasets. Their
attribution and separate dataset licenses are in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
CPU preparation receipts and source-readiness metadata are provenance, not
application results or substitutes for those notices.
