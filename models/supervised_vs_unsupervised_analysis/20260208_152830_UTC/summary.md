# Supervised vs Unsupervised Embedding Analysis Summary

- Timestamp: 20260208_152830_UTC
- Device: cuda
- Output root: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260208_152830_UTC

## Retrieval Ranking

              model_name  method_type  spearman  roc_auc   pr_auc       f1  accuracy
unsupervised_simcse_best unsupervised  0.492042 0.784076 0.826951 0.718182  0.713846
       all-mpnet-base-v2     baseline  0.462279 0.766892 0.813174 0.697385  0.626154
    supervised_v4_cosine   supervised  0.457240 0.763986 0.800624 0.669841  0.680000
        all-MiniLM-L6-v2     baseline  0.399127 0.730433 0.773240 0.699858  0.675385

## Winner

- Winner model: unsupervised_simcse_best
- Winner method: unsupervised
- Decision rule: retrieval_first_spearman_then_roc_auc_then_f1

## Artifact Paths

- Metrics JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260208_152830_UTC\metrics.json
- Cluster report CSV: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260208_152830_UTC\cluster_report.csv
- Cluster report JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260208_152830_UTC\cluster_report.json
- Semantic graphs: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260208_152830_UTC/semantic_graph_<model>.png