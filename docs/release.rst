Release notes
=============

v0.3.2
------

Initial public release of STORM (Spatial Temporal Omics Regulatory Modeling).

Highlights:

- Graph-linked unified embedding for paired multi-omics spatial data,
  derived from the GLUE framework with the following STORM-specific
  extensions:

  - Gene-program masking on the decoder (adapted from NicheCompass) so
    latent factors map onto interpretable biological programs.
  - Temporal-alignment objective for time-resolved samples.
  - Spatial-smoothed feature representations via per-modality k-NN graphs.

- User-facing API: :class:`storm.models.STORMModel`,
  :class:`storm.models.PairedSTORMModel`, and the high-level
  :func:`storm.models.fit_STORM` helper that runs the
  pretrain → balance → fine-tune loop.

- End-to-end six-notebook tutorial under ``examples/`` covering
  preprocessing, training, clustering, evaluation, gene-program
  activity, and gene-program spatial visualization.
