Tutorials
=========

The end-to-end STORM workflow is split across six notebooks that share a
common set of intermediate artifacts under ``./artifacts/storm_tutorial/``:

**Cell-embedding analysis** (``X_storm``)

1. :doc:`../examples/tutorial_1_preprocess` — load annotated raw AnnDatas,
   build prior gene-program masks, run spatial / Moran-I feature
   selection, construct the guidance graph, and emit the preprocessed
   AnnDatas.
2. :doc:`../examples/tutorial_2_train` — :func:`storm.models.fit_STORM`
   with :class:`storm.models.PairedSTORMModel`. Saves a ``.dill``
   checkpoint.
3. :doc:`../examples/tutorial_3_clustering` — three clustering modes on
   the shared latent space (modality-specific CONCAT vs.
   per-location JOINT) with biological interpretation and spatial / UMAP
   plots.
4. :doc:`../examples/tutorial_4_evaluation` — multi-omics integration
   (FOSCTTM, MLISI), cross-modality consistency (ARI), and cell-type
   recovery (NMI / ARI vs. ground-truth annotations).

**Gene-program activity analysis** (``X_storm_gp``)

5. :doc:`../examples/tutorial_5_gp_activity` — project cells into the
   GP-activity latent space, run per-cluster Wilcoxon enrichment, and
   draw cluster × GP heatmaps.
6. :doc:`../examples/tutorial_6_gp_visualization` — temporal GP analysis
   on a *separate* 5-timepoint dataset: temporal program modules
   (R1–R6 / A1–A8) and developmental-time × spatial-bin activity
   trajectories along the postnatal corpus callosum.

The raw annotated AnnDatas that tutorial 1 reads are provided under
``artifacts/storm_tutorial/raw/``. Tutorials 5 and 6 analyse specific
trained checkpoints whose matching inputs are staged by the
``examples/prepare_reftarg160_inputs.py`` and
``examples/prepare_5tmp_inputs.py`` helper scripts.
