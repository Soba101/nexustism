# Supervised vs Unsupervised Embedding Analysis Summary

- Timestamp: 20260212_080942
- Device: cuda
- Output root: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260212_080942

## Retrieval Ranking

              model_name  method_type  spearman  roc_auc   pr_auc       f1  accuracy
       all-mpnet-base-v2     baseline  0.483724 0.779273 0.822956 0.721585  0.618571
unsupervised_simcse_best unsupervised  0.480160 0.777216 0.822378 0.718157  0.702857
    supervised_v4_cosine   supervised  0.470885 0.771861 0.808336 0.686047  0.691429
        all-MiniLM-L6-v2     baseline  0.420434 0.742735 0.784770 0.708058  0.684286

## Winner

- Winner model: all-mpnet-base-v2
- Winner method: baseline
- Decision rule: retrieval_only_spearman_then_roc_auc_then_f1__hdbscan_unavailable

## Artifact Paths

- Metrics JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260212_080942\metrics.json
- Cluster report CSV: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260212_080942\cluster_report.csv
- Cluster report JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260212_080942\cluster_report.json
- Semantic graphs: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260212_080942/semantic_graph_<model>.png