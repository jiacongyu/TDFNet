# TDFNet : Tri-projection Deformable Fusion Network for 360° Salient Object Detection

Authors: Qiangqiang Zhou,Jiacong Yu,Jiawei Xu,Yong Chen,Xin Huang,Ping Li

------
# Introduction
In this work, we conduct 360° panoramic salient object detection by exploiting complementary projection representations to alleviate geometric distortions and improve detection performance. We propose TDFNet, the first Tri-projection Deformable Fusion Network for panoramic salient object detection. The key components are the cross-projection deformable attention (CDA) module and the latitude-guided fusion (LGF) module. CDA leverages spatial correspondences between different projections to construct geometry-aware sampling locations, guiding deformable attention for cross-projection contextual aggregation. LGF utilizes spherical latitude priors to construct geometric confidence weights for adaptively balancing ERP and CMP features, while incorporating distortion-reduced semantic references from Tangent Projection to achieve cross-projection feature refinement. By constructing a three-branch encoding architecture based on ERP, CMP, and Tangent Projection, TDFNet simultaneously preserves global spatial continuity, local geometric details, and fine-grained boundary information. Extensive experiments on four widely used PSOD benchmarks demonstrate the effectiveness of our method.

------
# Implementation

The source codes are available at [codes](https://github.com/jiacongyu/TDFNet).

The pretrained models and trained model weights of our TDFNet can be downloaded at [TDFNet-models](https://pan.baidu.com/s/1kUCwAS-LDH1d9YKHQzR5uA?pwd=9yb5).

The results of our TDFNet on 360-SOD, 360-SSOD, F-360iSOD, and ODI-SOD can be downloaded at [TDFNet-results](https://pan.baidu.com/s/1F-6sXFUj37scfEaayjoNjA?pwd=8sf2).

------
# Contact

E-mail address: jiacong_yu@jxnu.edu.cn
