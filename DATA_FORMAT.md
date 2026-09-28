# Input data format

SPARK-GRN expects one gene expression matrix and three edge splits. The full
multimodal model additionally requires a gene-level GeneActivity matrix.

## Matrices

- `ExpressionData.csv`: genes in rows, RNA cells in columns, with gene names in
  the first column.
- `GeneScoreData.csv`: genes in rows, ATAC cells in columns, with gene names in
  the first column. All RNA genes must be present; cell counts may differ.

## Edge splits

`Train_set.csv`, `Validation_set.csv`, and `Test_set.csv` must contain TF, TG,
and binary label columns. Accepted column names include `TF`/`TG`/`Label`,
`Gene1`/`Gene2`/`Label`, and `Source`/`Target`/`Score`. TF and TG values may be
gene names or zero-based indices into the expression matrix. The training prior
is constructed only from positive rows in `Train_set.csv`.

## Candidate file

Prediction accepts a CSV or delimiter-detected text file with two columns named
`TF` and `TG` (or `Gene1` and `Gene2`, or `Source` and `Target`). Values may be
gene names or zero-based indices.

## Batch-run directory layout

Paired and unpaired multi-omics runners use:

```text
Data/Multi-omics/
  dataset/<DATASET>/ExpressionData.csv
  dataset/<DATASET>/GeneScoreData.csv
  Train_validation_test_repeated/<DATASET>/Repeat_01/Fold_1/Train_set.csv
  Train_validation_test_repeated/<DATASET>/Repeat_01/Fold_1/Validation_set.csv
  Train_validation_test_repeated/<DATASET>/Repeat_01/Fold_1/Test_set.csv
```

The BEELINE runner uses the original fixed-fold layout:

```text
Data/BEELINE/
  dataset/<NETWORK>/<DATASET>/<TFs+500_or_TFs+1000>/BL--ExpressionData.csv
  Train_validation_test/<NETWORK>/<DATASET>/<SETTING>/Fold_1/Train_set.csv
  Train_validation_test/<NETWORK>/<DATASET>/<SETTING>/Fold_1/Validation_set.csv
  Train_validation_test/<NETWORK>/<DATASET>/<SETTING>/Fold_1/Test_set.csv
```

The processed matrices and fixed edge splits used by the supplied runners are
bundled in the layouts above. They are model-ready derivatives of public data,
not replacements for the original repository records. Source accessions and
links are reported in Supplementary Table S11, and preprocessing is described
in the Supplementary Methods.
