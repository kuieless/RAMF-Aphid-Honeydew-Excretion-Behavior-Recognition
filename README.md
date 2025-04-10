# Due to the complexity of the current codebase and environment, we are actively working on streamlining and optimizing the code.

<h2 align="center">
  RAMF-Aphid-Honeydew-Excretion-Behavior-Recognition
</h2>
<p align="center">
The codes、datasets、weights of paper《High-Throughput End-to-End Aphid Honeydew Excretion Behavior Recognition Method Based on Rapid Adaptive Motion-Feature Fusion》
</p>

<div align="center">
  <a href="http://www.shihuahuang.cn">Shihua Huang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=tIFWBcQAAAAJ&hl=en">Zhichao Lu</a><sup>2</sup>,
  <a href="https://vinthony.github.io/academic/">Xiaodong Cun</a><sup>3</sup>,
  Yongjun Yu<sup>1</sup>,
  Xiao Zhou<sup>4</sup>, 
  <a href="https://xishen0220.github.io">Xi Shen</a><sup>1*</sup>
</div>

## Dataset
In the experiment, the first fine-grained aphid behavior dataset, encompassing crawling, flicking, and honeydew excretion behaviors, was constructed.
Our dataset can be found at https://drive.google.com/drive/folders/1jJmJQ4SDzEU89UJcFL73Itk1t712wMut?usp=sharing

## Weights
The weight files mentioned in the paper can be found at https://drive.google.com/drive/folders/1IWmjOV7a7ilVVY_DsyyO_hXcQOXaDpBI?usp=sharing
https://drive.google.com/drive/folders/1x6HPU3UU2AmzN1T9mhmMuSibFtoCTB2J?usp=sharing

## Motion Feature
Use flow10.py to process all videos for generating datasets for annotation.
Use flow10%2.py for cross-frame processing of videos, where original frames and motion frames appear alternately. After processing, the video can be input for detection. (PS: When detecting aphids, we used cross-processing to detect Honeydew in the original frames; see the paper for specific details.)

## Detect
We have three versions for inference: the first two, detect2 and detect3, are slower in processing speed, while detect7 is the final version and the fastest.
For trained models, run detect2.py directly for real-time end-to-end streaming inference.
For complete pre-processing and post-processing, you need to replace the predictor.py in the engine directory, then run detect3 for streaming inference.
Run detect7-31.py to achieve real-time streaming inference at 31 fps.

## RAMF：Rapid Adaptive Motion-Feature Fusion
![UI Components Overview/UI组件界面](https://github.com/kuieless/RAMF-Rapid-Adaptive-Motion-Feature-Fusion/blob/main/RAMF.png)

## Stage-wise Honeydew Excreting Detection Pipeline and Results.
![UI Components Overview/UI组件界面](https://github.com/kuieless/RAMF-Rapid-Adaptive-Motion-Feature-Fusion/blob/main/Stage-wise%20Honeydew%20Excreting%20Detection%20Pipeline%20and%20Results.png)
