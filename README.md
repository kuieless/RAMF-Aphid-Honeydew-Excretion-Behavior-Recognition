# Note：Due to the complexity of the current codebase and environment, we are actively working on streamlining and optimizing the code.

<h2 align="center">
  RAMF-Aphid-Honeydew-Excretion-Behavior-Recognition
</h2>
<p align="center">
The codes、datasets、weights of paper《High-Throughput End-to-End Aphid Honeydew Excretion Behavior Recognition Method Based on Rapid Adaptive Motion-Feature Fusion》
</p>

<div align="center">
  Zhongqiang Song<sup>1</sup>,
  Jiahao Shen<sup>1</sup>, 
  Qiaoyi Liu<sup>1</sup>, 
  Wanyue Zhang<sup>1</sup>, 
  Ziqian Ren<sup>1</sup>, 
  Kaiwen Yang<sup>1</sup>, 
  Xinle Li<sup>1</sup>, 
  Jialei Liu<sup>1</sup>, 
  Fengming Yan<sup>1</sup>, 
  Wenqiang Li<sup>1</sup>, 
  Yuqing Xing<sup>1</sup>, 
  Lili Wu<sup>1</sup>, 
</div>

<p align="center">
<i>
1. College of Science, Henan Agricultural University, Zhengzhou, Henan 450002, China &nbsp; 2. College of Plant Protection, Henan Agricultural University, Zhengzhou, Henan 450046, China &nbsp; 3. College of Computing, City University of Hong Kong, Hong Kong, 999077, China &nbsp;
</i>
</p>

## 2. Quick start

### Setup
！python3.12.3
！PyTorch: 2.3.0+cu121
！torchvision: 0.18.0+cu121

```shell
conda create -n deim python=3.12.3
conda activate RAMF
pip install -r requirements.txt
```

### Weights
The weight files mentioned in the paper can be found at https://drive.google.com/drive/folders/1IWmjOV7a7ilVVY_DsyyO_hXcQOXaDpBI?usp=sharing
https://drive.google.com/drive/folders/1x6HPU3UU2AmzN1T9mhmMuSibFtoCTB2J?usp=sharing

### Dataset
In the experiment, the first fine-grained aphid behavior dataset, encompassing crawling, flicking, and honeydew excretion behaviors, was constructed.
Our dataset can be found at https://drive.google.com/drive/folders/

### Motion Feature
Use flow10.py to process all videos for generating datasets for annotation.
Use flow10%2.py for cross-frame processing of videos, where original frames and motion frames appear alternately. After processing, the video can be input for detection. (PS: When detecting aphids, we used cross-processing to detect Honeydew in the original frames; see the paper for specific details.)

### Detect
We have three versions for inference: the first two, detect2 and detect3, are slower in processing speed, while detect7 is the final version and the fastest.
For trained models, run detect2.py directly for real-time end-to-end streaming inference.
For complete pre-processing and post-processing, you need to replace the predictor.py in the engine directory, then run detect3 for streaming inference.
Run detect7-31.py to achieve real-time streaming inference at 31 fps.

### RAMF：Rapid Adaptive Motion-Feature Fusion
![UI Components Overview/UI组件界面](https://github.com/kuieless/RAMF-Rapid-Adaptive-Motion-Feature-Fusion/blob/main/RAMF.png)

### Stage-wise Honeydew Excreting Detection Pipeline and Results.
![UI Components Overview/UI组件界面](https://github.com/kuieless/RAMF-Rapid-Adaptive-Motion-Feature-Fusion/blob/main/Stage-wise%20Honeydew%20Excreting%20Detection%20Pipeline%20and%20Results.png)
