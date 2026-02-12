# Supervised vs Unsupervised Embedding Analysis Summary

- Timestamp: 20260209_073427
- Device: cuda
- Output root: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260209_073427

## Retrieval Ranking

                    model_name  method_type  spearman  roc_auc   pr_auc       f1  accuracy
          qwen3-embedding-0.6b     baseline  0.491339 0.783673 0.825462 0.694196  0.608571
             all-mpnet-base-v2     baseline  0.483724 0.779273 0.822956 0.721585  0.618571
      unsupervised_simcse_best unsupervised  0.480160 0.777216 0.822380 0.718157  0.702857
          supervised_v4_cosine   supervised  0.475915 0.774767 0.810363 0.714465  0.675714
multilingual-e5-large-instruct     baseline  0.463985 0.767878 0.819770 0.670947  0.707143
                        bge-m3     baseline  0.446452 0.757755 0.810963 0.681319  0.710000
              all-MiniLM-L6-v2     baseline  0.420434 0.742735 0.784851 0.708058  0.684286
         gte-multilingual-base     baseline  0.414565 0.739347 0.777636 0.639269  0.661429
       nomic-embed-text-v2-moe     baseline  0.412351 0.738069 0.790624 0.677878  0.644286
           embeddinggemma-300m     baseline  0.371892 0.714710 0.779824 0.666667  0.675714
             all-MiniLM-L12-v2     baseline  0.367511 0.712180 0.774838 0.626003  0.667143

## Winner

- Winner model: qwen3-embedding-0.6b
- Winner method: baseline
- Decision rule: retrieval_first_spearman_then_roc_auc_then_f1

## Artifact Paths

- Metrics JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260209_073427\metrics.json
- Cluster report CSV: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260209_073427\cluster_report.csv
- Cluster report JSON: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260209_073427\cluster_report.json
- Semantic graphs: C:\Users\donov\Documents\itsm project\Final_nexustism\nexustism\models\supervised_vs_unsupervised_analysis\20260209_073427/semantic_graph_<model>.png