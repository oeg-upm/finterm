# Offline validation of the anonymous source snapshot

- All Python source files compiled successfully; `python -m extractor --help` ran without model weights.
- `python -m unittest discover -s tests -q`: 180 tests, 173 passed and 7 skipped. Two of those skips are explicitly marked in this release: the original private byte-level integrity assertions are not applicable to anonymized files and omitted internal reports. Five other skips are inherited from the source suite.
- `python -m unittest discover -s functional_tests -q`: 18 passed and three test-module import errors because the corresponding historical `fixtures.py` was not present in the supplied archive. The missing fixtures were not fabricated.
- `python -m unittest discover -s reader_tests -q`: 44 passed and two failures in old source-layout/packaging invariants. These assertions expect an earlier source layout; they were retained unchanged.
- No GPU inference or external model download was performed. These results do not establish end-to-end equivalence with the archived experimental runs.

The manifest `SOURCE_MANIFEST.json` records SHA-256 checksums of this anonymous release, not those of the private historical repository. Prompt examples in the sibling directory retain their supplied message content verbatim.
