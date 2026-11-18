# Provenance of the final garlic benchmark inputs

On 18 November 2026, the project owner stated that they personally performed a human review of the **final recommendation file** and the **final 15-question annotation file** used for the garlic decision benchmark. This is an owner attestation recorded after the files were supplied; it is not a claim of independent double annotation or a substitute for a per-item review log.

The attestation is bound to these exact local files and SHA-256 hashes:

| Role | Local file, excluded from Git | SHA-256 |
| --- | --- | --- |
| Final 12-rule garlic S5 input | `data/private/verified_s5_rules.json` | `62fda994ddefae8dc6a5528fca2e9d327502811c1cb14a9a84487366a1e404db` |
| Final 15-question V5 reference annotations | `data/private/annotation_v5_gold.jsonl` | `aedf762f4d32e61f1332c4ecf9effc88d6ad169a97f99d4c52cec0245b2e9729` |

The V5 aggregate result references the same question-annotation hash. The S5 file hash matches the saved source manifest for the garlic benchmark. Neither file was changed to add this record.

The older R16 editing workspace and the empty, hash-named extraction gold templates for potato, garlic, and sesame are **different artifacts**. Their `UNREVIEWED` status does not describe the two final garlic benchmark inputs above. The owner attestation establishes the stated review provenance for those two files; the available files do not document a second reviewer, formal adjudication of disagreements, or an item-by-item review trail. Decision-class agreement also does not verify every generated dose, citation, or clarification text.
