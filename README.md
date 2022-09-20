# Lightweight Deep Network-Enabled Real-Time Low-Visibility Enhancement for Promoting Vessel Detection in Maritime Video Surveillance [Journal of Navigation]

[**[paper]**](https://www.researchgate.net/profile/Wen-Liu-41/publication/354542130_Lightweight_Deep_Network-Enabled_Real-Time_Low-Visibility_Enhancement_for_Promoting_Vessel_Detection_in_Maritime_Video_Surveillance/links/613ea85c01846e45ef44faff/Lightweight-Deep-Network-Enabled-Real-Time-Low-Visibility-Enhancement-for-Promoting-Vessel-Detection-in-Maritime-Video-Surveillance.pdf)

## 1. Requirement ##
* __Python__ == 3.7
* __Pytorch__ == 1.1.0

## 2. Abstract

Maritime video surveillance has become an essential part of the vessel traffic services system, which can guarantee vessel traffic safety and security in maritime applications. To make maritime surveillance more feasible and practicable, many intelligent vision-empowered technologies have been developed to automatically detect moving vessels from maritime visual sensing data (i.e., maritime surveillance videos). However, the visual data collected in low-visibility environment easily makes the essential optical information hidden in the dark, potentially resulting in decreased accuracy of vessel detection. To guarantee reliable vessel detection under lowvisibility conditions, we propose a low-visibility enhancement network (termed LVENet) based on Retinex theory to enhance imaging quality in maritime video surveillance. To be specific, LVENet is essentially a lightweight deep neural network by incorporating a depthwise separable convolution. The synthetically-degraded image generation and hybrid loss function are further presented to enhance the robustness and generalization capacities of our LVENet. Both full-reference and no-reference evaluation experiments have demonstrated that LVENet could yield comparable or even better visual qualities than other state-of-the-art methods. In addition, LVENet only needs 0.0045 seconds to restore degraded images of size 1920×1080 on an NVIDIA 2080Ti GPU, which can adequately meet real-time requirements. The vessel detection performance can also be tremendously promoted with the enhanced visibility under low-light imaging conditions.

## 3. Flowchart of Our Proposed Method

In this work, we design a lightweight convolutional neural network for learning the feature of maritime low-visibility scenes. In particular, we use depthwise separable convolution instead of traditional convolution to reduce model parameters and improve calculation speed. Give the current advances, no research has been conducted on the depthwise separable convolution adopted to low-light image enhancement thus far. Furthermore, a hybrid loss function is constructed to supervise the network training and enhance the network generalization. 

![Fig  2](https://user-images.githubusercontent.com/48637474/135222864-510ad3cb-2138-4182-bf67-84861d084e52.png)
**Figure 2. The flowchart of our proposed low-visibility enhancement network (LVENet). The DS-Conv and DS-DConv represent depthwise separable convolution and depthwise separable deconvolution, respectively.**

![Fig  3](https://user-images.githubusercontent.com/48637474/135223081-ce2cbf0b-8be1-46b1-8922-c1a9b37fbbb1.png)
**Figure 3. Usage case of traditional convolution and depthwise separable convolution.**
## 4. Training
* Put the Train images in the "data/train/syn" folder. [**[l6jq]**](https://pan.baidu.com/s/1u5qh5ipAwq5kGKVPlcw2_w)
* Put the Test image in the "data/input/syn" folder. [**[tqok]**](https://pan.baidu.com/s/1uokWPJWa6zwOT8ItWelVew)
* Run "data/img/prepare_patches.py" to generate the "train_syn.h5". 
* Modify the 145 lines of code in "main.py" to "parser.add_argument("--train", type=str, default =  True, help = '训练还是测试')".
* Run "main.py". 
* The trained model parameters will be saved in "checkpoint/". 

## 5. Testing
This code contains two modes, i.e., synthetic and real-world enhancement. 
### Synthetic Enhancement
* Put the low-light image in the "data/input/syn" folder.
* Modify the 145 lines of code in "main.py" to "parser.add_argument("--train", type=str, default =  False, help = '训练还是测试')".
* Modify the 154 lines of code in "main.py" to "parser.add_argument("--syn", type=str, default = True, help = 'syn or real')".
* Run "main.py". 
* The enhancement result will be saved in the "Result/output" folder.

### Real-World Enhancement
* Put the low-light image in the "data/input/real" folder.
* Modify the 145 lines of code in "main.py" to "parser.add_argument("--train", type=str, default =  False, help = '训练还是测试')".
* Modify the 154 lines of code in "main.py" to "parser.add_argument("--syn", type=str, default = False, help = 'syn or real')".
* Run "main.py". 
* The enhancement result will be saved in the "Result/output" folder.

#### If you have any questions, please get in touch with me (yuguo@whut.edu.cn).
