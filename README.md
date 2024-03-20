# HGTRDA
Exploring the association between ncRNAs and drug resistance is essential to unravel the molecular mechanisms of drug resistance, discover new drug targets or biomarkers, and optimise therapeutic regimens to improve treatment efficacy. Although traditional biological experiments are limited by their high cost and time consumption, the application of association prediction techniques has opened up new possibilities in the field of bioinformatics. However, the research in this field still needs to be deepened. To this end, this paper introduces the HGTRDA model, a novel computational framework that combines hypergraph neural network and topology-aware transformer techniques dedicated to predicting the potential association between ncRNAs and drug resistance.HGTRDA captures the topology-aware embedding of nodes via lightGCN coding and dynamically optimises the connections between nodes via hypergraph transformer to reveal implicit dependencies between nodes. The model is calibrated between local and global representation learning via self-supervised learning to correct for noise in the observed data, and uses an inner product approach to assess the degree of association between ncRNAs and drugs. Experiments on the ncRNADrug database show that HGTRDA outperforms the other five state-of-the-art methods in predicting ncRNA-drug resistance associations. Case studies also validated the effectiveness and potential application of HGTRDA as a prediction tool.

# Requirements
- tensorflow-gpu 1.15.0
- python 3.6.15
- numpy 1.17.2
- pandas 1.1.5
- scikit-learn 0.22.1
- scipy 1.5.4

# Data
ncRNADrug\citealp{cao2024ncrnadrug} is an integrated and comprehensive database that contains information on ncRNA-drug resistance, ncRNA-drug targets, and potential drug combinations for treating drug-resistant cancers, derived from manual curation and computational inference. Currently, ncRNADrug includes 29,551 experimentally validated ncRNA-drug resistance associations, covering 9,195 ncRNAs (2,248 miRNAs, 4,145 lncRNAs, and 2,802 circRNAs) and 266 drugs, as well as 624,246 ncRNA-drug resistance associations predicted based on ncRNA expression profiles, covering 134,201 ncRNAs (including 3,601 miRNAs, 32,892 lncRNAs, and 97,708 circRNAs) and 5,588 drugs.
# Run the demo
```
python Main.py
```
