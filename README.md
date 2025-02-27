# PAM-Denoising
Paper: Score-based diffusion priors-guided generative deep learning for highly robust 3D denoising in photoacoustic microscopy      
Authors: Xianlin Song, Yiguang Wang, Xuelin Liu, Yubin Cao, Chenxi Tang, Yue Lei, Guolin Liu, Jiawen hu, Weiliang Yuan, Siyi Cao, Qiegen Liu       
Optics & Laser Technology       

Date : Feb-24-2025     
Version : 1.0       
The code and the algorithm are for non-comercial use only.      
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.     
      
Photoacoustic microscopy (PAM) is characterized by high resolution, high contrast, and deep tissue penetration, making it extensively utilized in the field of biomedical imaging. However, due to factors such as laser pulse energy fluctuations, external environmental interference, and system noise, a significant amount of noise is introduced into the photoacoustic signal during the imaging process. In this research, a denoising approach for PAM images based on the score-based diffusion model is introduced. During the training stage, the diffusion model is employed to learn the prior distribution of noise-free images. In the reconstruction phase, the learned priors serve as constraints to effectively denoise the input PAM images. Specifically, a penalized weighted least-squares (PWLS) term is incorporated into the iterative process to enhance denoising performance. After multiple iterations and solutions, PAM images with a low noise level are reconstructed and generated. The simulation results indicate that, even at a high noise level (e.g., 16 dB), the proposed method achieves average Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) values of 35.2 dB and 0.943, respectively. Furthermore, the experimental results indicate that the Contrast-to-Noise Ratio (CNR) of the proposed method is 2.31 times higher than that of the fully dense U-Net (FD-U-Net) model, highlighting its superior denoising performance and enhanced capability to reconstruct high-quality three-dimensional spatial images. It confirms the feasibility of this method for practical applications. This method also provides the possibility for low dose PAM.  

## Method.
<div align="center"><img src="https://github.com/yqx7150/PAM-3D-Denoising/blob/main/Fig3.tif"> </div>

Fig. 3. Flowchart of denoising iterative reconstruction based on the score-based diffusion model.       
    
## Results on simulation data.
<div align="center"><img src="https://github.com/yqx7150/PAM-3D-Denoising/blob/main/Fig5.tif"> </div>

Fig. 5. The reconstruction process of B-scan images of randomly distributed points and blood vessels using the proposed method. 

<div align="center"><img src="https://github.com/yqx7150/PAM-3D-Denoising/blob/main/Fig6.tif"> </div>

Fig. 6. The contrast experiments of the FD-U-Net and score-based diffusion models for randomly distributed points are shown as follows. 

<div align="center"><img src="https://github.com/yqx7150/PAM-3D-Denoising/blob/main/Fig7.tif"> </div>

Fig. 7. The contrast experiments of the FD-U-Net and score-based diffusion models for randomly distributed points are shown as follows. 

## Results on phantom experiment data.
<div align="center"><img src="https://github.com/yqx7150/PAM-3D-Denoising/blob/main/Fig8.tif"> </div>

Fig. 8.   Reconstruction results for the resolution test target and tungsten wire. 

## Other Related Projects
* Sparse-view reconstruction for photoacoustic tomography combining diffusion model with model-based iteration      
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597923001118)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-Diffusion)

* Score-based generative model-assisted information compensation for high-quality limited-view reconstruction in photoacoustic tomography      
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597924000405)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Limited-view-PAT-Diffusion)
    
* Ultra-sparse reconstruction for photoacoustic tomography: sinogram domain prior-guided method exploiting enhanced score-based diffusion model      
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597924000879)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-Sinogram-Diffusion)

* Mean-reverting diffusion model-enhanced acoustic-resolution photoacoustic microscopy for resolution enhancement: Toward optical resolution      
[<font size=5>**[Paper]**</font>](https://doi.org/10.1142/S1793545824500238)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/https://github.com/yqx7150/PAM-AR2OR)

* Unsupervised disentanglement strategy for mitigating artifact in photoacoustic tomography under extremely sparse view      
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S2213597924000302?via%3Dihub)       [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-ADN)
      
* Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
      
* Generative model for sparse photoacoustic tomography artifact removal      
[<font size=5>**[Paper]**</font>](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12745/1274503/Generative-model-for-sparse-photoacoustic-tomography-artifact-removal/10.1117/12.2683128.short?SSO=1)               
     
* PAT-public-data from NCU [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PAT-public-data)
