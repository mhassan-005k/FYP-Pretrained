Development of Super Resolution Algorithm for enhancement of High

Resolution Satellite Images

## Project Proposal

## Supervisor

## Dr. Saad Akbar

## Co-supervisor

## Miss Aqsa Nadir

## Submitted by

Muhammad Hassan

2505-2023

Muhammad Mahad Khan

2197-2023

## Department of Computer Science,

Hamdard University, Karachi.

\_________________


## 1. Introduction

Satellites take pictures of the Earth from space, but because of the extreme distance and the atmosphere, there is a limit to how clear these images can be. When we try to zoom in past the camera's original limit, the images become blurry and blocky [1]. Our project aims to address this using a process called "Super Resolution."

Super Resolution uses Artificial Intelligence to try to enhance the spatial resolution of satellite imagery beyond the sensor’s original limits. It attempts to reconstruct the fine, pixel-level details that would be impossible to capture with the satellite’s native camera [2]. In collaboration with SUPARCO, we aim to develop a robust AI-based super-resolution algorithm solution for satellite imagery. To achieve this, we will explore either combining multiple pre- trained models into a unified framework or fine-tuning a high-performing pre-trained model, selecting the approach that delivers the best performance for SUPARCO’s specific use case. . While fulfilling SUPARCO's requirement for a working software tool, we will also try to improve the efficiency and performance of the chosen AI framework so we can publish two strong research papers during our FYP 1 and FYP 2 phases.

## 2. Objective

The main goal of this project is to build and deploy an AI-powered software tool that attempts to make satellite images sharper and clearer. To achieve this, we have set the following 3 objectives:

- To explore and evaluate different AI techniques (such as Convolutional Neural Networks, Generative Adversarial Networks, or modern frameworks) to identify the most suitable approach for satellite imagery [3].

- To try to develop an AI algorithm that aims to improve image resolution without faking or inventing false geographic details.

- To try to optimize the efficiency of the chosen model framework and deploy it as a user- friendly software tool for SUPARCO analysts.

## 3. Problem Description

When standard software tries to enlarge a picture, it simply stretches the existing pixels, which creates a blurry image. AI can address this by learning what the missing details should look like and filling them in. However, applying AI to satellite imagery comes with two major challenges that we plan to try to solve:

First, AI sometimes "hallucinates." This means it might invent details to make the picture look good, like adding a fake road or a building that does not exist in real life [4]. For SUPARCO, the


maps must be accurate. We need to try to ensure our model remains factual while sharpening edges.

Second, satellite image files are massive. Running advanced AI on these huge files can quickly run out of memory and crash a standard system [5]. A major part of our research will be finding ways to try to make the final algorithm lightweight and memory-efficient so it works reliably in a real-world deployed setting.

## 4. Comprehensive Literature Review

The primary goal of this research is to try to develop or optimize a super-resolution tool that is both accurate and visually sharp. Early methods focused heavily on standard Convolutional Neural Networks (CNNs) [6]. While CNNs are mathematically stable, they often produce slightly blurred edges. To fix this blurring, researchers introduced Generative Adversarial Networks (GANs) which create very sharp textures [7], though they carry a risk of hallucinating details [8].

More recent studies have tried to balance this by using dynamic routing networks to process different parts of the image based on complexity [9]. Advanced architectures like Vision Transformers [10] and continuous State Space Models (Mamba) [11] have also been explored to understand macro-level geographic structures without running out of memory. Finally, Diffusion Probabilistic Models [12] have shown promise in highly detailed image recovery. Our project will explore these methods, ensuring our deployed tool can handle real-world spatial degradations effectively [13], [14], [15].

## 4.1 Table: Summary of Related Work

| Referenc e | Author & Year | Methodology / Architecture | Findings (Key Contribution) | Limitations (Gap Addressed by Proposed Project) |
| --- | --- | --- | --- | --- |
| [1] | Ma et al. (2020) | Comprehensiv e Survey / Review | Identified core challenges in SISR for remote sensing, including SWaP constraints and blind degradation. | A survey paper; does not propose a specific novel architectural solution. |
| [2] | Dong et al. (2016) | Shallow CNN (SRCNN) | Pioneered the use of deep learning for image super- resolution, | Produces overly smoothed images; struggles with high- frequency spatial |


|   |   |   | establishing a mathematically stable baseline. | details in complex maps. |
| --- | --- | --- | --- | --- |
| [3] | Zhang et al. (2018) | Residual Dense Network (RDN) | Leveraged residual dense blocks to extract abundant local features, improving reconstruction quality. | High computational complexity and memory usage; difficult to scale for massive satellite tiles. |
| [4] | Ledig et al. (2017) | Generative Adversarial Network (SRGAN) | Introduced perceptual loss to reconstruct photo- realistic high- frequency textures. | High risk of hallucinating artificial structures, violating factual accuracy needed in GIS. |
| [5] | Li et al. (2022) | Neural Architecture Search (NAS) | Automated the design of memory- efficient architectures for image restoration tasks. | Search process is highly computationally expensive; resulting models may still lack global geographic context. |
| [6] | Wang et al. (2022) | Lightweight CNN Optimization | Significantly reduced computer memory usage while keeping the structural layout accurate. | Relies purely on local processing; struggles to make fine urban details perfectly clear. |
| [7] | Jiang et al. (2019) | Edge- Enhanced GAN | Utilized edge- enhancement to sharpen boundaries in remote sensing mapping. | Still requires careful balancing to avoid introducing artificial noise into homogeneous regions. |
| [8] |   | Jia et al. Multiattention | Created very sharp, | High risk of |


|   | (2022) | Generative Model | highly detailed images by focusing on visual realism and textures. | hallucinating fake details, making it unstable for strict geographic tasks. |
| --- | --- | --- | --- | --- |
| [9] | Wu et al. (2023) | Dynamic Routing Framework | Made the AI smarter by having it spend less compute power on smooth areas (like water). | Still fundamentally suffers from the over-smoothing effect common in traditional approaches. |
| [10] | Hou et al. (2024) | Vision Transformers (Swin) | Extracted global features across large images without running out of memory using phase learning. | Still requires substantial computational processing power for continuous satellite tiles. |
| [11] | Zhi et al. (2024) | State Space Modeling (Mamba) | Combined lightweight memory usage with high-quality edge sharpness using linear sequence math. | Needs a reliable software wrapper and extensive testing to be fully deployed for analysts. |
| [12] | Xiao et al. (2024) | Diffusion Probabilistic Model | Modeled fine- grained detailed recovery step-by- step for real-world blind degradation. | Iterative denoising steps cause high inference latency, requiring speed- based code enhancement. |
| [13] | Gu et al. (2021) | Swin Transformer (SwinIR) | Replaced standard CNNs with shifted- window transformers, achieving state-of- the-art structural alignment. | High memory consumption when applied to large, unbroken remote sensing data arrays. |


| [14] | Cornillere et al. (2019) | Blind Super- Resolution Framework | Successfully modeled spatially variant blur and degradation, common in satellite optics. | The estimation of blur kernels is mathematically unstable on highly complex terrain textures. |
| --- | --- | --- | --- | --- |
| [15] | Haut et al. (2018) | Deep Generative Network | Tailored generative adversarial techniques specifically for multi-spectral remote sensing data. | Early generative approach prone to severe mode collapse and structural distortion if untrained properly. |

## 5. Methodology

Iterative Refinement Based on Results

*Fig 5.1 Methodology overview of the proposed framework.*

To align with our FYP schedule and meet SUPARCO's project requirements, we will carry out this research in five progressive steps. This structured workflow allows us to establish immediate baselines using existing tools before moving on to our own model development and performance optimizations.


- Step 1: Data Collection and Preprocessing We will try to use datasets consisting of SUPARCO's own satellite images. However, if those are not confirmed or available early on, we will use other open-source benchmark satellite images (such as UC Merced or AID) to supplement our training. We will take clear images and apply simulated blur and noise to create a paired baseline dataset.

- Step 2: Establishing Baselines We will first run standard pre-trained models (such as CNNs, SRGAN, or SwinIR) out of the box. This provides an immediate performance benchmark for SUPARCO and documents the limitations of current state-of-the-art algorithms on raw satellite data.

- Step 3: Custom Development and Optimization Using baseline insights, we will develop our enhanced approach by fine-tuning a strong model or combine them (like Mamba or Diffusion). To handle massive satellite files without memory crashes, we will implement an automated image tiling and stitching pipeline.

- Step 4: Evaluation and Comparison We will compare our custom, optimized model against the step 2 baselines using quantitative metrics (ex ; PSNR, SSIM) alongside qualitative checks to ensure no geographic features are falsely generated. This data will form the core of our research publications.

- Step 5: Software Packaging and Deployment Finally, we will build a web interface this will allow SUPARCO analysts to upload raw satellite images, process them using our optimized pipeline, and download the enhanced high-resolution results.

## 6. Project Scope

## Scope In

- Develop and train an AI algorithm for Single Image Super Resolution (SISR).

- Processing and handling massive, high-resolution satellite imagery (sourced from SUPARCO or open benchmarks).

- Optimizing the algorithm for memory efficiency using techniques like image chunking/tiling.

- Deploying the finalized AI model into a functional software tool for end-users.

## Scope Out

- Live, real-time processing of active orbital video streams.

- Modification or enhancement of Synthetic Aperture Radar (SAR) data.

- Development of fully offline, standalone desktop applications beyond the required software tool deployment scope.

- Hardware-level satellite system design or sensor-level modifications.


## 7. Feasibility Study

## 7.1 Risks and Mitigation

| Risk | How We Plan to Handle It |
| --- | --- |
| System runs out of memory (OOM) | We will write our code to automatically cut massive satellite images into smaller tiles, process them independently, and stitch them back together at the end. |
| The AI invents fake details (Hallucinations) | We will incorporate strict structural checks during training to force the model to try to respect factual geography over purely aesthetic sharpness. |
| Dataset constraints or delay | Since the exact dataset from SUPARCO is not confirmed yet, our ingestion pipeline will be built flexibly, allowing us to easily toggle between datasets. |
| Training runs out of daily time limits | We will implement robust state checkpointing scripts, saving our training progress automatically so we can resume immediately without data loss. |

## 7.2 Resource Requirements

- Compute: A computer equipped with a dedicated graphics card (GPU), minimum RTX 4090 with 16 GB of VRAM, for running, optimizing, and eventually deploying the AI frameworks.

- Storage: 50GB to 100GB of storage space allocated for satellite dataset management and model weight storage.

- Datasets: Target satellite imagery pairs (to be sourced from SUPARCO or open benchmark repositories).

## 8. Solution Application Areas

This deployed software tool will be directly useful for SUPARCO's GIS and Image Processing departments. Analysts can use it to try to get a clearer view of urban infrastructure,


agricultural lands, or environmental changes without needing to buy expensive new satellite hardware. Beyond SUPARCO, this research can help any industry that relies on aerial mapping, such as drone surveying, forestry management, or disaster relief mapping.

## 9. Tools / Technology

| Tool / Technology | Purpose |
| --- | --- |
| Python / C++ / Matlab | Languages for core algorithmic logic and image processing tasks, meeting SUPARCO prerequisites. |
| PyTorch / TensorFlow | Deep learning frameworks used to evaluate, construct, or enhance the AI models. |
| OpenCV / Rasterio | Basic image manipulation, applying synthetic degradation, and handling metadata. |
| Deployment Frameworks (e.g., FastAPI/Streamlit) | Deploying the final algorithm into an accessible, standalone software tool interface. |
| GitHub | Code repository management, version control, and team collaboration. |

## 10. Responsibilities of Team Members

R = Responsible | A = Accountable | C = Consulted | I = Informed

| Task | Muhammad Hassan | Muhammad Mahad Khan |
| --- | --- | --- |
| Literature Review & Research | R, A | R, A |
| Data Gathering & Tiling Setup | R,A | R,C, A |


| Model Evaluation & Assembly | R, A | R, A |
| --- | --- | --- |
| Efficiency Optimization Tweaks | R,C, A | R, A |
| Software Tool Deployment | R, A | R, A |
| Final Tool Testing & Validation | R, A | R, A |
| Research Paper Drafting (FYP 1) | R, A | R,A |
| Final Findings Documentation (FYP 2) | R,A | R, A |

## 11. Planning & Milestones

| Phase | Task | Assigned To | Timeline |
| --- | --- | --- | --- |
| Phase 1: FYP 1 Focus | Project Kickoff, Data Gathering, & Baseline Testing | Hassan, Mahad | Month 1 - 2 |
|   | SUPARCO Milestone 1: Literature survey. Identifying existing super resolution tools and techniques. | Hassan, Mahad | Month 3 |
|   | FYP 1 Deliverable: Draft and submit FYP 1 Research Paper (Literature Review). | Hassan, Mahad | Month 3 |
| Phase 2: FYP 2 Focus | Algorithmic Evaluation, Experimental training, & Model Fusion/Selection | Hassan, Mahad | Month 4 - 6 |
|   | SUPARCO Milestone 2: Tool Development and Integration. Develop and demonstrate initial model. | Hassan, Mahad | Month 7 |


|   | Heavy Pipeline Optimization, Validation, & Paper Writing | Hassan, Mahad | Month 8 - 11 |
| --- | --- | --- | --- |
|   | SUPARCO Milestone 3: Validation and Final Reporting. Successful testing on high-res satellite imagery. | Hassan, Mahad | Month 12 |
|   | FYP 2 Deliverable: Draft and submit FYP 2 Research Paper (Findings and Optimization Solutions). | Hassan, Mahad | Month 12 |

## Key Project Deliverables:

- Software Tool: A fully deployed application for image resolution enhancement.

- Software Code: The complete, documented source code containing our models and optimization logic.

- Research Papers: Two academic papers (Literature Survey for FYP 1, and Findings/Improvements for FYP 2).

## 12. References

- [1] W. Ma et al., "Remote sensing image super-resolution: Challenges and approaches," IEEE Geoscience and Remote Sensing Magazine, 2020.

- [2] C. Dong et al., "Image super-resolution using deep convolutional networks," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.

- [3] Y. Zhang et al., "Residual dense network for image super-resolution," Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

- [4] C. Ledig et al., "Photo-realistic single image super-resolution using a generative adversarial network," Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

- [5] Y. Li et al., "Memory-efficient hierarchical neural architecture search for image restoration," International Journal of Computer Vision, 2022.

- [6] X. Wang et al., "Contextual Transformation Network for Lightweight Remote-Sensing Image Super-Resolution," IEEE Transactions on Geoscience and Remote Sensing, 2022.

- [7] K. Jiang et al., "Edge-enhanced GAN for remote sensing image superresolution," IEEE Transactions on Geoscience and Remote Sensing, 2019.


[8] X. Jia et al., "Multiattention Generative Adversarial Network for Remote Sensing Image Super-Resolution," ISPRS Journal of Photogrammetry and Remote Sensing, 2022.

[9] Y. Wu et al., "Lightweight Stepless Super-Resolution of Remote Sensing Images via Saliency- Aware Dynamic Routing Strategy," arXiv preprint arXiv:2210.07598, 2023.

[10] J. Hou et al., "CSwT-SR: Conv-Swin Transformer for Blind Remote Sensing Image Super- Resolution With Amplitude-Phase Learning," IEEE Transactions on Geoscience and Remote Sensing, 2024.

[11] L. Zhi et al., "MambaFormerSR: A Lightweight Model for Remote-Sensing Image Super- Resolution," IEEE International Geoscience and Remote Sensing Symposium, 2024.

[12] Y. Xiao et al., "EDiffSR: An Efficient Diffusion Probabilistic Model for Remote Sensing Image Super-Resolution," IEEE Transactions on Geoscience and Remote Sensing, 2024.

[13] J. Gu et al., "SwinIR: Image restoration using swin transformer," Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

[14] V. Cornillere et al., "Blind image super-resolution with spatially variant degradations," ACM Transactions on Graphics, 2019.

[15] J. M. Haut et al., "A new deep generative network for remote sensing image super-

resolution," Remote Sensing, 2018.
