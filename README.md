
# Improving Detection Accuracy and Efficiency: A Comprehensive Evaluation of YOLOv11 Integrated with the Object Guided Inference Slicing Framework on Small Object Detection Challenges [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](OGIS-Det_VS_FI-Det(Evaluation).ipynb)

## Description
*(Small Object Detection (SOD) remains a persistent challenge in computer vision, particularly in applications involving high-resolution imagery, such as UAV-based monitoring, medical diagnostics, and autonomous navigation. Issues like scale variation, occlusion, and background clutter complicate the detection of small objects, often requiring innovative approaches. This paper introduces the Object Guided Inference Slicing (OGIS) framework, a novel and adaptive two-stage slicing strategy designed to optimize computational resource allocation and improve detection accuracy for small and medium objects. Integrated with the YOLOv11 architecture, OGIS leverages dynamic coarse-to-fine slicing to prioritize Regions of Interest (ROIs), enhancing detection precision without compromising efficiency.Empirical evaluations on the VisDrone2019 dataset reveal the transformative impact of OGIS, achieving a 468.08% increase in AP@[small] and a 735.41% improvement in AR@[small], alongside a 27.05% reduction in inference time compared to traditional Full Inference Detection (FI-Det). Additionally, OGIS outperforms established methods, such as Slicing Aided Hyper Inference (SAHI) and Entropy-Based Region Prioritization, in critical metrics like precision, recall, and F1 score. These results highlight the framework's capability to balance detection performance across small, medium, and large objects, ensuring robust scalability and adaptability for real-world applications.While the framework demonstrates remarkable performance, trade-offs are observed in large object detection due to the prioritization of dense regions for small object detection. Future research will focus on refining the slicing strategy to dynamically balance performance across all object scales, as well as exploring energy-efficient adaptations and cross-dataset evaluations to enhance generalizability. With its state-of-the-art results, OGIS sets a new benchmark for resource-aware and adaptive small object detection methodologies.. )*

---
## Testing Code Steps

### 1. **Download Required Files**
- **Ground Truth (GT)**: Download the COCO.json file containing the ground truth annotations.
- **FI-Det COCO.json**: Download the Full Inference Detection results in COCO.json format.
- **OGIS-Det COCO.json**: Download the Object Guided Inference Slicing Detection results in COCO.json format.
- Upload the files to your preferred storage location (e.g., Google Drive).

### 2. **Setup Environment**
Clone the repository and install necessary dependencies:
```bash
git clone https://github.com/muza-g/SOD_YOLOv11_OGIS_Benchmarks-.git
pip install pycocotools
```
### 3. **Evaluation Steps**
Run the following commands to evaluate and compare results:

#### **FI-Det Evaluation**
```bash
python Results_Evaluate.py \
    --ground_truth_path ./data/ground_truth/ground_truth_coco.json \
    --predictions_path ./data/FI_Predictions/full_inference.json
```

#### **OGIS-Det Evaluation**
```bash
python Results_Evaluate.py \
    --ground_truth_path ./data/ground_truth/ground_truth_coco.json \
    --predictions_path ./data/OGIS_Predictions/gois_inference.json
```

#### **Comparison of FI-Det and OGIS-Det**
```bash
python Compare_Results_OGIS_Det_VS_FI_Det.py \
    --ground_truth_path ./data/ground_truth/ground_truth_coco.json \
    --full_inference_path ./data/FI_Predictions/full_inference.json \
    --gois_inference_path ./data/OGIS_Predictions/gois_inference.json
```


## Part 1: NOT Fine-Tuned (NFT) YOLOv10, Results are saved in **COCO.json** format.  
**Inference Experiment on 100% VisDrone2019 Train Dataset (6,471 Images)**
-  [**Ground Truth (GT)**:](https://drive.google.com/file/d/1-xQ6z7Yx0y3pZp_TZpbWjbM4VrbQ4yL0/view?usp=drive_link)  - [**Full Inference Detection (FI-Det)**:](https://drive.google.com/file/d/1UY5555KzgbNu2Ao4LWKLE2WDNyNZb2Zx/view?usp=drive_link)  -  [**Object Guided Inference Slicing Detection (OGIS-Det)**:](https://drive.google.com/file/d/1-9mf-KK9sFkcvMx-RN7NWxA7lsuTFrV6/view?usp=drive_link)

## Evaluation Table: YOLO11 FI-Det vs. OGIS-Det  

The following table compares **Full Inference Detection (FI-Det)** and **Object Guided Inference Slicing Detection (OGIS-Det)** for YOLO11, presented in a four-row structure.

| **Metric**               | **FI-Det** | **OGIS-Det** | **% Improvement** |
|---------------------------|------------|--------------|--------------------|
| **AP@[IoU=0.50:0.95]**    | 0.008      | 0.020        | ↑ 154.65%         |
| **AP@[IoU=0.50]**         | 0.012      | 0.030        | ↑ 163.22%         |
| **AP@[IoU=0.75]**         | 0.008      | 0.021        | ↑ 161.41%         |
| **AP@[small]**            | 0.001      | 0.007        | ↑ 468.08%         |
| **AP@[medium]**           | 0.014      | 0.034        | ↑ 146.81%         |
| **AP@[large]**            | 0.041      | 0.060        | ↑ 45.21%          |
| **AR@[IoU=0.50:0.95|max=1]** | 0.007   | 0.016        | ↑ 127.93%         |
| **AR@[IoU=0.50:0.95|max=10]**| 0.016  | 0.039        | ↑ 144.30%         |
| **AR@[IoU=0.50:0.95|max=100]**| 0.017 | 0.050        | ↑ 191.32%         |
| **AR@[small]**            | 0.003      | 0.021        | ↑ 735.41%         |
| **AR@[medium]**           | 0.028      | 0.083        | ↑ 197.72%         |
| **AR@[large]**            | 0.075      | 0.120        | ↑ 60.16%          |

---

## Observations  

- **Significant Improvements in Small Object Detection**:  
  The OGIS-Det framework exhibited a dramatic increase in **AP@[small]** (+468.08%) and **AR@[small]** (+735.41%), demonstrating its strength in detecting small objects.  

- **Enhanced Recall**:  
  Substantial improvements in **AR@[IoU=0.50:0.95|max=100]** (+191.32%) and **AR@[medium]** (+197.72%) highlight OGIS-Det’s effectiveness in comprehensive object detection.  

- **Improved Localization Accuracy**:  
  The increases in **AP@[IoU=0.50:0.95]** (+154.65%) and **AP@[IoU=0.50]** (+163.22%) validate OGIS-Det's ability to enhance detection precision across IoU thresholds.  

- **Balanced Performance Across Object Sizes**:  
  While excelling in small and medium object detection, OGIS-Det also achieved notable gains in **AP@[large]** (+45.21%) and **AR@[large]** (+60.16%), ensuring robust detection across all scales.  

---


## Part 2: NOT Fine-Tuned (NFT) YOLO11 
**Inference Experiment on 15% VisDrone2019 Train Dataset (970 Images) AND Available Data:Results are saved in **COCO.json** format
-  [**Ground Truth (GT)**:](https://drive.google.com/file/d/1kFNr8s_Yg0Yb0xxXuF4awaIWCwpehGTZ/view?usp=drive_link)  [**Full Inference Detection (FI-Det)**:](https://drive.google.com/file/d/1-wNxSIM0XV5fP0vp_9vhdY4RtTKq2yMo/view?usp=drive_link)   [**Object Guided Inference Slicing Detection (OGIS-Det)**:](https://drive.google.com/file/d/14JKhfzgQP8KUFS_Ek0jLxLLsMchREE2B/view?usp=drive_link)

## Evaluation Table: YOLO11 FI-Det vs. OGIS-Det

The following table compares **Full Inference Detection (FI-Det)** and **Object Guided Inference Slicing Detection (OGIS-Det)** for YOLO11, along with the percentage improvement achieved by OGIS-Det over FI-Det.

| **Metric**           | **mAP-Small** | **AR-Small** | **mAP-Medium** | **mAP-Large** | **AR@1** | **AR@10** | **AR@100** | **AR-Medium** | **AR-Large** | **F1 Score** | **mAP@0.50:0.95** | **mAP@0.50** | **mAP@0.75** |
|-----------------------|---------------|---------------|----------------|---------------|-----------|-----------|------------|---------------|--------------|--------------|-------------------|-------------|--------------|
| **FI-Det**           | 0.02          | 0.04          | 0.23           | 0.57          | 0.12      | 0.27      | 0.29       | 0.49          | 1.09         | 0.17         | 0.12              | 0.18        | 0.13         |
| **OGIS-Det**         | 0.10          | 0.33          | 0.57           | 0.96          | 0.27      | 0.68      | 0.87       | 1.40          | 1.93         | 0.47         | 0.33              | 0.51        | 0.34         |
| **% Improve**        | ↑ 400.0%      | ↑ 725.0%      | ↑ 147.83%      | ↑ 68.42%      | ↑ 125.0%  | ↑ 151.85% | ↑ 200.0%   | ↑ 185.71%     | ↑ 77.06%     | ↑ 176.47%    | ↑ 175.0%          | ↑ 183.33%   | ↑ 161.54%    |

---

### Observations  

- **Significant Improvements**: The OGIS-Det framework demonstrated remarkable performance enhancements over FI-Det, with standout improvements in **AR-Small** (+725.0%) and **mAP-Small** (+400.0%), confirming its effectiveness in detecting small objects.  

- **Enhanced Detection Accuracy**: Substantial gains in **mAP@0.50** (+183.33%) and **mAP@0.75** (+161.54%) indicate that OGIS-Det consistently provides better localization accuracy across various IoU thresholds.  

- **Superior Recall Performance**: The results highlight significant recall improvements, particularly in **AR@100** (+200.0%) and **AR-Medium** (+185.71%), showcasing OGIS-Det's ability to comprehensively detect objects, especially those of medium size.  

- **Balanced Performance Across Scales**: While the framework excelled in small and medium object detection, it also achieved notable gains in **mAP-Large** (+68.42%) and **AR-Large** (+77.06%), ensuring robust performance across different object sizes.  

- **Efficient Resource Utilization**: The observed enhancements in F1 Score (+176.47%) demonstrate that OGIS-Det achieves a balanced trade-off between precision and recall, making it highly suitable for real-world scenarios requiring efficient yet accurate object detection.  

---

## Part 3: Fine-Tuned (FT) YOLO11
**Trained for 10 Epochs on 100% VisDrone2019 Train Dataset (6,471 Images) and Inference on the Same Dataset, Results are saved in **COCO.json** format.**
- [**Ground Truth (GT)**: ](https://drive.google.com/file/d/15KjnH9FoEfxb9ZnIOjPPMwGsPgxC7Yfu/view?usp=drive_link)  [ **Full Inference Detection (FI-Det)**:](https://drive.google.com/file/d/1BRUjIj2ZfQgdU0rN7TNlEmcQ_ts1A6_k/view?usp=drive_link)   [**Object Guided Inference Slicing Detection (OGIS-Det)**:](https://drive.google.com/file/d/1-7it__F-imlERoMa5D8j0JxolY1MmP4x/view?usp=drive_link)


### Evaluation Table: YOLO11 FI-Det vs. OGIS-Det

The following table compares **Full Inference Detection (FI-Det)** and **Object Guided Inference Slicing Detection (OGIS-Det)** for YOLO11, along with the percentage improvement achieved by OGIS-Det over FI-Det.

### Evaluation Table: YOLO11 FI-Det vs. OGIS-Det  

The following table provides a comparative analysis of **Full Inference Detection (FI-Det)** and **Object Guided Inference Slicing Detection (OGIS-Det)** for YOLO11, presented in a four-row structure.

| **Metric**           | **AP-Small** | **AR-Small** | **AP-Medium** | **AP-Large** | **AR@1** | **AR@10** | **AR@100** | **AR-Medium** | **AR-Large** | **F1 Score** | **AP@[IoU=0.50:0.95]** | **AP@[IoU=0.50]** | **AP@[IoU=0.75]** |
|-----------------------|--------------|--------------|---------------|--------------|-----------|-----------|------------|---------------|--------------|--------------|-------------------------|-----------------|-------------------|
| **FI-Det**           | 0.024        | 0.035        | 0.159         | 0.283        | 0.045     | 0.112     | 0.137      | 0.208         | 0.349        | 0.170        | 0.120                   | 0.171           | 0.119            |
| **OGIS-Det**         | 0.071        | 0.133        | 0.164         | 0.151        | 0.053     | 0.152     | 0.207      | 0.273         | 0.227        | 0.470        | 0.134                   | 0.192           | 0.132            |
| **% Improve**        | ↑ 196.90%    | ↑ 278.66%    | ↑ 2.94%       | ↓ 46.71%     | ↑ 18.81%  | ↑ 35.46%  | ↑ 51.17%   | ↑ 31.44%      | ↓ 34.90%     | ↑ 176.47%    | ↑ 12.01%                | ↑ 12.38%        | ↑ 11.26%         |


---
### Observations  

- **Significant Improvements in Small Object Detection**: OGIS-Det exhibited substantial enhancements in detecting small objects, with **AP-Small** increasing by 196.90% and **AR-Small** improving by 278.66% compared to FI-Det.  

- **Enhanced Recall Performance**: OGIS-Det demonstrated remarkable recall improvements, particularly in **AR@100** (+51.17%) and **AR-Medium** (+31.44%), highlighting its ability to effectively capture medium-sized objects in cluttered environments.  

- **Balanced Detection Accuracy**: While OGIS-Det showed modest gains in **AP-Medium** (+2.94%), it achieved significant improvements in **F1 Score** (+176.47%), indicating better precision-recall trade-offs.  

- **Trade-Offs in Large Object Detection**: A slight decline in **AP-Large** (-46.71%) and **AR-Large** (-34.90%) was observed, reflecting OGIS-Det’s prioritization of small and medium objects over large ones, aligning with its adaptive resource allocation strategy.  

- **Overall Efficiency**: Improvements in key metrics such as **AP@[IoU=0.50]** (+12.38%) and **AP@[IoU=0.75]** (+11.26%) further validate OGIS-Det’s capability to achieve enhanced localization accuracy while maintaining computational efficiency.  

---

## Citation  
If you use this framework or dataset in your research, please cite this work:  

```bibtex
@article{yourpaper2025,
  title={Improving Detection Accuracy and Efficiency: A Comprehensive Evaluation of YOLOv11 Integrated with the Object Guided Inference Slicing Framework on Small Object Detection Challenges},
  author={Your Name and Co-Authors},
  journal={Preprint},
  year={2025},
  note={Available at [insert-link-to-preprint]}
}
```

---

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

---

## Acknowledgements  
We extend our heartfelt gratitude to the collaborative research team at the College of Computer Science and Technology, Zhejiang University, China, for their invaluable support and guidance.  

---

## Contact  
For inquiries or collaborations, please contact:  
**Email:** muzamal@zju.edu.cn  
```

