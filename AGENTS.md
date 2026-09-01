# REVOLT Copper Simulation Agent Instructions

This package owns the REVOLT Copper instrument simulation. It depends on
AdaptiveOpticsSim.jl for reusable physics, algorithms, graph execution, and
HIL boundaries.

- Keep REVOLT Copper configuration, calibration provenance, graph files,
  Pyramid-sensor settings, integration code, and instrument validation here.
- Do not duplicate generic AdaptiveOpticsSim algorithms in this package.
- Keep provisional and inferred instrument values explicitly labeled and
  separate from measured calibration.
- Preserve exact HSDM277 command and detector-frame order at external
  boundaries.
- Use immutable configuration and prepared execution owners; preallocate
  repeated-frame storage.
- Use explicit seeded RNG ownership for deterministic validation.
- Validate CPU locally first. Use the local AMDGPU host and WSL CUDA only for
  changes that affect accelerator execution.
- Commit generated manifests only in a deliberately maintained application or
  validation environment, not at the package root.
