# Export validation

Checks performed on this exported source package, not the development checkout:

- Imports resolve to the exported `neurocad` directory.
- 66 focused offline tests passed. Seven upstream `ezdxf`/Pyparsing deprecation warnings were observed.
- The procedural demo generated one request with 16 STEP candidates.
- The frozen seed-1701 checkpoint produced eight ranked hypotheses on that request.
- The first proposal passed the demo's pair and global geometry checks, and the assembly STEP was exported and re-import checked by the export utility.
- The prediction stage did not read the generated reference targets. No live LLM request was made.
- All three checkpoint files match the recorded source freeze and contain tensor-only state dictionaries.

This smoke run establishes software operability for the supplied example. It does not establish general task accuracy or reproduce the full paper benchmark. Generated demo outputs and local console logs are excluded from the source release.

Validation environment: Python 3.12, the recorded Windows CadQuery/OCP environment, and PyTorch. Tests use generated fixtures, including accepted and rejected assembly cases.
