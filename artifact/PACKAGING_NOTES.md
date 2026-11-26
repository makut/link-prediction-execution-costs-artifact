# Current v0.6.0 packaging scope

The published v0.5.0 release and six companion assets remain immutable. The current v0.6.0 payload adds independently verified E22/E23 recorded evidence and E24 posthoc analysis. Original creation records and overwritten active metadata are preserved under `metadata/lineage/BEFORE_V0_6_FINALIZATION/`. The current complete inventory is `MANIFEST.json`; whole code-archive SHA and publication receipts remain external to avoid a self-checksum cycle. Git/archive/release actions are separate from this metadata stage.

## Historical packaging/license record

# Candidate packaging and license scope

The user approved MIT for the authors’ own code on 25 November 2026. This includes
the new public adapters and the authors’ PENCIL/native bridge code. Historical
frozen source notes that say its own-code license was undecided describe an
earlier state; they are preserved without changing the measured source bytes.

PENCIL commit 2d32e29dbed533288d9d758138e07547a0a7d8a9 is Apache-2.0. Its complete
tracked source, exact LICENSE and existing copyright notices are retained in
work/vldb_pencil_bridge/upstream_pencil. No upstream root NOTICE exists at that
pin. The new bridge changes the physical induced-subgraph operator; upstream
consumer/model sources remain unchanged and their verifier checks every file.
Google BERT config attribution, source/model-card hashes and Apache-2.0
declaration are retained in bert_config/source.json; no pretrained weights.

The native standalone control retains the torch-sparse MIT text, pinned
saint_cpu.cpp source and adaptation notes. Existing SEAL_OGB and VK-GNN notices
remain in the base package. MIT for own code does not replace those notices.

Friendster fixed-S records and E16 query/timing records derive only from the
public densely relabelled Zenodo 5146230 graph, CC-BY-4.0; retain the attribution
in DATA_ACCESS.md. No private VK fixed-S report or replay plan is included.

SYSTEMS_FROZEN_INPUTS.json pins unmodified experiment sources and public reference
inputs. MANIFEST.json describes the complete locally frozen payload; the prior
working-candidate inventory is retained separately by the packager. New adapters
are distinct from frozen measured sources. Local archive creation is separate
from external publication, which has not been performed.
