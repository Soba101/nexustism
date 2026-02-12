# Supervised vs Unsupervised Embedding Analysis Summary

- Timestamp: 20260210_063940
- Device: cuda
- Output root: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260210_063940

## Retrieval Ranking

                    model_name  method_type  spearman  roc_auc   pr_auc       f1  accuracy
          supervised_v4_cosine   supervised  0.501249 0.789392 0.830283 0.724607  0.624286
             all-mpnet-base-v2     baseline  0.483724 0.779273 0.822956 0.721585  0.618571
      unsupervised_simcse_best unsupervised  0.480160 0.777216 0.822378 0.718157  0.702857
multilingual-e5-large-instruct     baseline  0.463993 0.767886 0.819531 0.670947  0.707143
                        bge-m3     baseline  0.446452 0.757755 0.810963 0.681319  0.710000
              all-MiniLM-L6-v2     baseline  0.420434 0.742735 0.784770 0.708058  0.684286
           embeddinggemma-300m     baseline  0.371891 0.714710 0.779692 0.666667  0.675714

## Winner

- Winner model: supervised_v4_cosine
- Winner method: supervised
- Decision rule: retrieval_first_spearman_then_roc_auc_then_f1

## Artifact Paths

- Metrics JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260210_063940\metrics.json
- Cluster report CSV: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260210_063940\cluster_report.csv
- Cluster report JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260210_063940\cluster_report.json
- Semantic graphs: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260210_063940/semantic_graph_<model>.png