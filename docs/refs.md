```
title: U-net: Convolutional networks for biomedical image segmentation
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

```
title: Deep residual learning for image recognition
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

```
title: Variational Bayesian model selection for mixture distributions
@inproceedings{corduneanu2001variational,
  title={Variational Bayesian model selection for mixture distributions},
  author={Corduneanu, Adrian and Bishop, Christopher M},
  booktitle={Proceedings eighth international conference on artificial intelligence and statistics},
  pages={27--34},
  year={2001},
  organization={Morgan Kaufmann}
}
```

```
title: Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?
@article{bernard2018deep,
  title={Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Clement and Cervenansky, Frederick and Yang, Xin and Heng, Pheng-Ann and Cetin, Irem and Lekadir, Karim and Camara, Oscar and Ballester, Miguel Angel Gonzalez and others},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={11},
  pages={2514--2525},
  year={2018},
  publisher={ieee}
}
```

```
title: Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M&Ms challenge
@article{campello2021multi,
  title={Multi-centre, multi-vendor and multi-disease cardiac segmentation: the M\&Ms challenge},
  author={Campello, Victor M and Gkontra, Polyxeni and Izquierdo, Cristian and Martin-Isla, Carlos and Sojoudi, Alireza and Full, Peter M and Maier-Hein, Klaus and Zhang, Yao and He, Zhiqiang and Ma, Jun and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={12},
  pages={3543--3554},
  year={2021},
  publisher={IEEE}
}
```

```
title: Deep Gaussian mixture model for unsupervised image segmentation

abstract: The recent emergence of deep learning has led to a great deal of work on designing supervised deep semantic segmentation algorithms. As in many tasks sufficient pixel-level labels are very difficult to obtain, we propose a method which combines a Gaussian mixture model (GMM) with unsupervised deep learning techniques. In the standard GMM the pixel values with each sub-region are modelled by a Gaussian distribution. In order to identify the different regions, the parameter vector that minimizes the negative log-likelihood (NLL) function regarding the GMM has to be approximated. For this task, usually iterative optimization methods such as the expectation-maximization (EM) algorithm are used. In this paper, we propose to estimate these parameters directly from the image using a convolutional neural network (CNN). We thus change the iterative procedure in the EM algorithm replacing the expectation-step by a gradient-step with regard to the networks parameters. This means that the network is trained to minimize the NLL function of the GMM which comes with at least two advantages. As once trained, the network is able to predict label probabilities very quickly compared with time consuming iterative optimization methods. Secondly, due to the deep image prior our method is able to partially overcome one of the main disadvantages of GMM, which is not taking into account correlation between neighboring pixels, as it assumes independence between them. We demonstrate the advantages of our method in various experiments on the example of myocardial infarct segmentation on multi-sequence MRI images.

@inproceedings{schwab2024deep,
  title={Deep Gaussian mixture model for unsupervised image segmentation},
  author={Schwab, Matthias and Mayr, Agnes and Haltmeier, Markus},
  booktitle={International Conference on Machine Learning, Optimization, and Data Science},
  pages={339--352},
  year={2024},
  organization={Springer}
}
```

```
title: Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation

abstract: Semi-supervised segmentation remains challenging in medical imaging since the amount of annotated medical data is often scarce and there are many blurred pixels near the adhesive edges or in the low-contrast regions. To address the issues, we advocate to firstly constrain the consistency of pixels with and without strong perturbations to apply a sufficient smoothness constraint and further encourage the class-level separation to exploit the low-entropy regularization for the model training. Particularly, in this paper, we propose the SS-Net for semi-supervised medical image segmentation tasks, via exploring the pixel-level smoothness and inter-class separation at the same time. The pixel-level smoothness forces the model to generate invariant results under adversarial perturbations. Meanwhile, the inter-class separation encourages individual class features should approach their corresponding high-quality prototypes, in order to make each class distribution compact and separate different classes. We evaluated our SS-Net against five recent methods on the public LA and ACDC datasets. Extensive experimental results under two semi-supervised settings demonstrate the superiority of our proposed SS-Net model, achieving new state-of-the-art (SOTA) performance on both datasets.

@inproceedings{wu2022exploring,
    title={Exploring Smoothness and Class-Separation for Semi-supervised Medical Image Segmentation},
    author={Wu, Yicheng and Wu, Zhonghua and Wu, Qianyi and Ge, Zongyuan and Cai, Jianfei},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={34--43},
    volume={13435},
    year={2022},    
    doi={10.1007/978-3-031-16443-9\_4},
    organization={Springer, Cham}
}
```

```
title: Decoupled Consistency for Semi-supervised Medical Image Segmentation

abstract: By fully utilizing unlabeled data, the semi-supervised learning (SSL) technique has recently produced promising results in the segmentation of medical images. Pseudo labeling and consistency regularization are two effective strategies for using unlabeled data. Yet, the traditional pseudo labeling method will filter out low-confidence pixels. The advantages of both high- and low-confidence data are not fully exploited by consistency regularization. Therefore, neither of these two methods can make full use of unlabeled data. We proposed a novel decoupled consistency semi-supervised medical image segmentation framework. First, the dynamic threshold is utilized to decouple the prediction data into consistent and inconsistent parts. For the consistent part, we use the method of cross pseudo supervision to optimize it. For the inconsistent part, we further decouple it into unreliable data that is likely to occur close to the decision boundary and guidance data that is more likely to emerge near the high-density area. Unreliable data will be optimized in the direction of guidance data. We refer to this action as directional consistency. Furthermore, in order to fully utilize the data, we incorporate feature maps into the training process and calculate the loss of feature consistency. A significant number of experiments have demonstrated the superiority of our proposed method.

@InProceedings{10.1007/978-3-031-43907-0_53,
author="Chen, Faquan
and Fei, Jingjing
and Chen, Yaqi
and Huang, Chenxi",
editor="Greenspan, Hayit
and Madabhushi, Anant
and Mousavi, Parvin
and Salcudean, Septimiu
and Duncan, James
and Syeda-Mahmood, Tanveer
and Taylor, Russell",
title="Decoupled Consistency for Semi-supervised Medical Image Segmentation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="551--561",
abstract="By fully utilizing unlabeled data, the semi-supervised learning (SSL) technique has recently produced promising results in the segmentation of medical images. Pseudo labeling and consistency regularization are two effective strategies for using unlabeled data. Yet, the traditional pseudo labeling method will filter out low-confidence pixels. The advantages of both high- and low-confidence data are not fully exploited by consistency regularization. Therefore, neither of these two methods can make full use of unlabeled data. We proposed a novel decoupled consistency semi-supervised medical image segmentation framework. First, the dynamic threshold is utilized to decouple the prediction data into consistent and inconsistent parts. For the consistent part, we use the method of cross pseudo supervision to optimize it. For the inconsistent part, we further decouple it into unreliable data that is likely to occur close to the decision boundary and guidance data that is more likely to emerge near the high-density area. Unreliable data will be optimized in the direction of guidance data. We refer to this action as directional consistency. Furthermore, in order to fully utilize the data, we incorporate feature maps into the training process and calculate the loss of feature consistency. A significant number of experiments have demonstrated the superiority of our proposed method. The code is available at https://github.com/wxfaaaaa/DCNet.",
isbn="978-3-031-43907-0"
}
```

```
title: Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation

abstract: Consistency learning is a central strategy to tackle unlabeled data in semi-supervised medical image segmentation (SSMIS), which enforces the model to produce consistent predictions under the perturbation. However, most current approaches solely focus on utilizing a specific single perturbation, which can only cope with limited cases, while employing multiple perturbations simultaneously is hard to guarantee the quality of consistency learning. In this paper, we propose an Adaptive Bidirectional Displacement (ABD) approach to solve the above challenge. Specifically, we first design a bidirectional patch displacement based on reliable prediction confidence for unlabeled data to generate new samples, which can effectively suppress uncontrollable regions and still retain the influence of input perturbations. Meanwhile, to enforce the model to learn the potentially uncontrollable content, a bidirectional displacement operation with inverse confidence is proposed for the labeled images, which generates samples with more unreliable information to facilitate model learning. Extensive experiments show that ABD achieves new state-of-the-art performances for SSMIS, significantly improving different baselines. 

@inproceedings{chi2024adaptive,
  title={Adaptive Bidirectional Displacement for Semi-Supervised Medical Image Segmentation},
  author={Chi, Hanyang and Pang, Jian and Zhang, Bingfeng and Liu, Weifeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4070--4080},
  year={2024}
}
```

```
title: LSRL-Net: A level set-guided re-learning network for semi-supervised cardiac and prostate segmentation

abstract: Semi-supervised medical image segmentation reduces the reliance on extensive labeled datasets, which is particularly crucial in the medical field. However, existing mean teacher (MT) models often encounter challenges with inaccurate predictions when dealing with organs that possess complex anatomical structures or indistinct boundaries, such as the heart and prostate. Specifically, for organs with complex morphology and irregular boundaries, both teacher and student models may generate similar segmentation errors, resulting in the accumulation of inaccurate information throughout the training process, which ultimately affects the segmentation accuracy. To address these issues, this paper proposes a novel semi-supervised learning framework, termed LSRL-Net, which is based on a level set-guided relearning network. Firstly, a level set module is utilized for the pre-segmentation of medical images, producing an initial segmentation contour that guides the teacher model and enhances its performance in regions with ambiguous boundaries. Secondly, under the MT framework, a re-learning module is implemented to retrain the inaccurate data predicted by both the teacher and student models, preventing the spread and accumulation of wrong information and enhancing the stability of the model in training. Experimental evaluations conducted on the ACDC and PROMISE12 datasets demonstrate the superior performance of LSRL-Net in cardiac and prostate segmentation tasks, achieving SOTA performance metrics. The proposed framework improves segmentation accuracy for organs with complex structures and unclear boundaries.

@article{LIU2025108062,
title = {LSRL-Net: A level set-guided re-learning network for semi-supervised cardiac and prostate segmentation},
journal = {Biomedical Signal Processing and Control},
volume = {110},
pages = {108062},
year = {2025},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2025.108062},
url = {https://www.sciencedirect.com/science/article/pii/S1746809425005737},
author = {Ruihua Liu and Jiangyu Liao and Xinyu Liu and Yanwei Liu and Yijie Chen},
keywords = {Level set method, Re-learning module, Mean teacher model, Medical image segmentation, Semi-supervised learning},
abstract = {Semi-supervised medical image segmentation reduces the reliance on extensive labeled datasets, which is particularly crucial in the medical field. However, existing mean teacher (MT) models often encounter challenges with inaccurate predictions when dealing with organs that possess complex anatomical structures or indistinct boundaries, such as the heart and prostate. Specifically, for organs with complex morphology and irregular boundaries, both teacher and student models may generate similar segmentation errors, resulting in the accumulation of inaccurate information throughout the training process, which ultimately affects the segmentation accuracy. To address these issues, this paper proposes a novel semi-supervised learning framework, termed LSRL-Net, which is based on a level set-guided relearning network. Firstly, a level set module is utilized for the pre-segmentation of medical images, producing an initial segmentation contour that guides the teacher model and enhances its performance in regions with ambiguous boundaries. Secondly, under the MT framework, a re-learning module is implemented to retrain the inaccurate data predicted by both the teacher and student models, preventing the spread and accumulation of wrong information and enhancing the stability of the model in training. Experimental evaluations conducted on the ACDC and PROMISE12 datasets demonstrate the superior performance of LSRL-Net in cardiac and prostate segmentation tasks, achieving SOTA performance metrics. The proposed framework improves segmentation accuracy for organs with complex structures and unclear boundaries.}
}
```

```
title: beta-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation

abstract: Co-training has achieved significant success in the field of semi-supervised learning(SSL); however, the homogenization phenomenon, which arises from multiple models tending towards similar decision boundaries, remains inadequately addressed. To tackle this issue, we propose a novel algorithm called β-FFT from the perspectives of data processing and training structure. In data processing, we apply diverse augmentations to input data and feed them into two sub-networks. To balance the training instability caused by different augmentations during consistency learning, we introduce a nonlinear interpolation technique based on the Fast Fourier Transform (FFT). By swapping low-frequency components between variously augmented images, this method not only generates smooth and diverse training samples that bridge different augmentations but also enhances the model's generalization capability while maintaining consistency learning stability. In training structure, we devise a differentiated training strategy to mitigate homogenization in co-training. Specifically, we use labeled data for additional training of one model within the co-training framework, while for unlabeled data, we employ linear interpolation based on the Beta (β) distribution as a regularization technique in additional training. This approach allows for more efficient utilization of limited labeled data and simultaneously improves the model's performance on unlabeled data, optimizing overall system performance.

@inproceedings{hu2025beta,
  title={beta-FFT: Nonlinear Interpolation and Differentiated Training Strategies for Semi-Supervised Medical Image Segmentation},
  author={Hu, Ming and Yin, Jianfu and Ma, Zhuangzhuang and Ma, Jianheng and Zhu, Feiyu and Wu, Bingbing and Wen, Ya and Wu, Meng and Hu, Cong and Hu, Bingliang and others},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30839--30849},
  year={2025}
}
```

```
title: Adaptive Learning of High-Value Regions for Semi-Supervised Medical Image Segmentation

abstract: Existing semi-supervised learning methods typically mitigate the impact of unreliable predictions by suppressing low-confidence regions. However, these methods fail to explore which regions hold higher learning value and how to design adaptive learning strategies for these regions. To address these issues, we propose a novel adaptive learning of high-value regions (ALHVR) framework. By exploiting the diversity of predictions from dual-branch networks, the prediction regions are classified into three groups: reliable stable region, reliable unstable region, and unreliable stable region. For high-value regions (reliable unstable region and unreliable stable region), different training strategies are designed. Specifically, for reliable unstable region, we propose a confidence-guided cross-prototype consistency learning (CG-CPCL) module, which enforces prototype consistency constraints in the feature space. By leveraging confidence information, the high-confidence predictions from one network selectively supervise the low-confidence predictions from the other, thus helping the model learn inter-class discrimination more stably. Additionally, for unreliable stable region, we design a dynamic teacher competition teaching (DTCT) module, which dynamically selects the most reliable pixels as teachers by evaluating the unperturbed predictions from both networks. These selected pixels are then used to supervise perturbed predictions, thereby enhancing the model's learning capability in unreliable region. Experimental results show that our method outperforms state-of-the-art approaches on three public datasets.

@inproceedings{lei2025adaptive,
  title={Adaptive Learning of High-Value Regions for Semi-Supervised Medical Image Segmentation},
  author={Lei, Tao and Yang, Ziyao and Wang, Xingwu and Wang, Yi and Wang, Xuan and Sun, Feiman and Nandi, Asoke K},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21450--21459},
  year={2025}
}
```