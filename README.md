# DDENet
## Abstract
Breast cancer is one of the most common types of cancer affecting women worldwide. Ultrasound imaging is a safe and cost-effective diagnostic modality that has been widely adopted for early diagnosis. However, breast ultrasound image segmentation faces several specific challenges, including the mismatch between global semantics and local structural details, the semantic gap between the encoder and decoder, and insufficient sensitivity to multi-scale heterogeneity and blurred boundaries. Therefore, a Dual-Domain Enhanced Network with Multi-scale Structure Guide (DDENet) is designed for lesion segmentation in breast ultrasound images. Firstly, a Multi-Branch Downsampling module (MBD) is introduced, which integrates multi-path downsampling, dynamic feature fusion, and fine-grained enhancement to effectively preserve key information often lost in standard downsampling operations. Secondly, a Fourier-Enhanced Deformable block (FED) is proposed to perform dual-domain enhancement of structure-aware and high-order semantic features in the spatial and frequency domains. This improves the encoder’s ability to model regions of different scales with blurred boundaries. To bridge the inherent semantic gap between the deep-level semantics of the encoder and the shallow-level structure of the decoder, an Adaptive Matching Channel Modulation (AMCM) and a Reverse Attention Fusion (RAF) module are strategically embedded into the bottleneck layer. Specifically, the former accentuates key semantic regions through adaptive channel reconstruction, while the latter explicitly recovers boundary structures via a reverse attention mechanism, ensuring the effective fusion of deep and shallow features. Finally, at the decoding stage, a Multi-Scale Structure Guide module (MSG) is developed through combining multi-scale feature extraction with a local-global gating mechanism, thereby enabling the decoder to achieve progressive reconstruction of fine-grained structural details and contextual information of breast lesions. Experimental results demonstrate that our DDENet achieves Dice scores of 83.63%, 91.58%, and 84.73% on the BUSI, BUSBRA, BrEaST datasets, respectively, which are higher than the other comparison methods. In addition, extensive cross-dataset validation experiments further verify the model’s exceptional generalizability.
## Datesets: 
https://pan.baidu.com/s/1by24xKiueewxl0XVb0pfEw 提取码: hadh
## Citations
If you find this work helpful, please cite:

```bibtex
@article{ma2026dual,
  title={A dual-domain enhanced network with multi-scale structure guide for breast lesion segmentation in ultrasound images},
  author={Ma, Changlong and Li, Haiyan and Liu, Yajie and Shi, Xin and He, Bingbing and Liu, Xiang},
  journal={Expert Systems with Applications},
  pages={132176},
  year={2026},
  publisher={Elsevier}
}
```
