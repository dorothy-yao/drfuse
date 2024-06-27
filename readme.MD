# DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency (AAAI2024)
This directory contains source codes of `DrFuse: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency` by AAAI-24.

## Abstract
The combination of electronic health records (EHR) and medical images is crucial for clinicians in making diagnoses and forecasting prognoses. Strategically fusing these two data modalities has great potential to improve the accuracy of machine learning models in clinical prediction tasks. However, the asynchronous and complementary nature of EHR and medical images presents unique challenges. Missing modalities due to clinical and administrative factors are inevitable in practice, and the significance of each data modality varies depending on the patient and the prediction target, resulting in inconsistent predictions and suboptimal model performance. To address these challenges, we propose DrFuse to achieve effective clinical multi-modal fusion. It tackles the missing modality issue by disentangling the features shared across modalities and those unique within each modality.

Furthermore, we address the modal inconsistency issue via a disease-wise attention layer that produces the patient- and disease-wise weighting for each modality to make the final prediction. We validate the proposed method using real-world large-scale datasets, MIMIC-IV and MIMIC-CXR. Experimental results show that the proposed method significantly outperforms the state-of-the-art models.

![Overview](https://github.com/dorothy-yao/drfuse/blob/main/overview.png "overview_framework")
**Overview**：DrFuse consists of two major components. Subfigure (a): A shared representation and a distinct representation are learned from EHR and CXR, where the shared ones are aligned by minimizing the Jensen–Shannon divergence (JSD). A novel logit pooling is proposed to fuse the shared representations. Subfigure (b): The \textit{disease-aware attention fusion} module captures the patient-specific modal significance for different prediction targets by minimizing a ranking loss.

## Acknowledgements
Some parts of the codes are adapted from [MedFuse](https://github.com/nyuad-cai/MedFuse). We thank the authors for their work. 

## Reproduction
To reproduce the results, please follow the following procedures:

- Obtain [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) datasets;
- Follow the data pre-processing procedures in https://github.com/nyuad-cai/MedFuse to prepare the datasets
- Run the code by `python main.py <ARGUMENTS>`

## Citation
If you find the paper or the implementation helpful, please cite the following paper:

```bib
@inproceedings{yao2024drfuse,
  title={{DrFuse}: Learning Disentangled Representation for Clinical Multi-Modal Fusion with Missing Modality and Modal Inconsistency},
  author={Yao, Wenfang and Yin, Kejing and Cheung, William K and Liu, Jia and Qin, Jing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={16416--16424},
  year={2024}
}
```
