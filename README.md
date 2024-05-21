# Deepfake_Detection

- 논문링크 추후에

# 프로젝트 소개

Our 베이스라인 모델 : [Real forensics](https://github.com/ahaliassos/RealForensics)

위 베이스라인 모델은 딥페이크 탐지를 위해 Student-Teacher 프레임워크를 사용했으며, 비디오 데이터에는 CSN모델, 오디오 데이터에 대해서는 ResNet 모델로 표현학습을 사용합니다.  

저희는 공간적 정보를 잘 추출하는 기존 CNN기반의 CSN 모델에서 시간적 정보를 잘 추출하기 위해 비디오 데이터 학습모델의 백본을 변경하고, 대용량의 비디오 데이터셋이 아닌 5000개의 데이터 만으로 좋은 성능을 낼 수 있는 모델을 설계합니다.

# CGATT Backbone
<img width="672" alt="스크린샷 2024-05-21 오후 10 19 54" src="https://github.com/ta-ho/CGATT/assets/127817503/2ed8e089-accb-42f0-aa7a-374faa848a81">

저희의 새로운 백본인 CGATT 입니다. 3DCNN과 GRU,Attention 블록을 사용하여 비디오 데이터의 시공간적 정보를 효과적으로 추출할 수 있도록 합니다. 

- **`3D Convolutional Network`  :** Video 데이터(H*W*F(frames))를 처리 할 수 있는 CNN을 사용합니다.
    - Channel수는 RGB인 3으로 시작하지만, 3dConv 를 거쳐 256개의 표현으로 늘어나고 최종적으로 GRU에 입력하기 위해 flatten을 사용합니다.
- **`Gate recurrent Unit (GRU)` :** 시간적 정보를 효과적으로 추출하기 위해 시퀀스데이터 계열의 모델인 GRU 사용합니다.
    - GRU는 LSTM(Long Short-Term Memory)과 비슷한 기능을 가지고 있지만, 더 간단한 구조로 시간적 데이터의 패턴 및 특징을 파악하고 다음 프레임의 예측에 활용할 수 있습니다.
    - 따라서 딥페이크 영상의 시간적 일관성을 보다 정확하게 분석하고 탐지할 수 있습니다.
- **`Attention` :** 시간 차원에 **중요도 가중치**를 계산하여 비디오 시퀀스에서 특정 시간 단계의 특성들의 중요도를 높이기 위해 Attention 모듈을 사용합니다.
    - 각 시간 단계에서의 특성에 대한 가중치를 동적으로 할당하여 모델이 더 집중해야 할 중요한 프레임을 강조합니다.
    

# 성능
![Untitled (1)](https://github.com/ta-ho/CGATT/assets/127817503/e8e218ba-8764-4808-b62d-7121d3f3cb19)


CGATT모델을 통하여 기존 모델보다 ssl_weight 를 0.3으로 조정하였을 때, 가장 좋은 결과가 나올 수 있었습니다. FaceShifter, CelebDf 에서 모두 기존의 CSN 을 이용한 성능보다 더 높게 나옴을 확인할 수 있습니다.

저희는 가장 높게 나온 성능의 CGATT 체크포인트 파일을 제공합니다.
