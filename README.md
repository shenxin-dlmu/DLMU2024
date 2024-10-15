# DLMU2024
Underwater detection dataset <br>

通过网盘分享的文件：DLMU_2024.zip <br>
链接: https://pan.baidu.com/s/1It1zaZsCuwzsZP5N24Zm9g  <br>
提取码: dmrp <br>

If you use this underwater dataset, please cite this github link and our paper as follows: <br>
https://github.com/shenxin-dlmu/DLMU2024 <br>
This paper is being published: **Unsupervised Clustering Optimization-based Efficient Visual Perception Attention in YOLO for Underwater Object Detection**<br>

The main contributions of our work are summarized as follows:<br>

* We explore the optimal attention design suitable for underwater object detection, and propose an unsupervised clustering optimization-based effcient attention (UCOEA) to reduce underwater background interference and improve underwater object perception. Our design strategies can better balance additional cost overhead and information processing quality, which iscrucial for the proposed attention to achieve fast and accurate underwaterinformation calibration.<br>

* We design a channel clustering strategy, which achieves autonomousdyamic screening of channel information by using the K-Means algorithm. Same types of channel information with high redundancy are learneduniformly to share the same operation. Different types of channel information with high specifcity are learned independently to avoid channel noise information interference.<br>

* We design a spatial clustering strategy, which achieves autonomous dynamicstripping of spatial information by using the EM algorithm. This strategycan extract multiple required spatial information at one time from different spatial locations. We further assign learnable weight parameters to distinguish dominant information and auxiliary information, which can alleviate spatial noise information interference.<br>

* In order to achieve high-precision and real-time underwater object detection,we propose a combined system of UCOEA underwater adapter and one-stage YOLO detector, which can efciently detect small, medium and largetargets at the same time. The effectiveness of our work is comprehensively verified on 22 popular attention modules, 2 advanced YOLO detectors with 3 diferent sizes, and 5 public underwater datasets.<br>

* We collect,produce and publish an underwater detection dataset DLMU2024 with low image continuity and high data diversity, which provides reliable support for the rapid development of underwater detection research. Our dataset and code are available at https://github.com/shenxin-dlmu/DLMU2024.<br>

