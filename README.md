### SNU DL Study Entrance Test
#### Animal Image Classification

MobileNet V2 모델을 이용하여 전이학습을 수행하였고, 모델의 최상위 레이어에 대하여 Fine-Tuning도 진행하여 정확도를 향상시켰습니다


RMSPropOptimizer과 Crossentropy Loss Function을 사용하였습니다.

Train/Valid/Test 비율은 8:1:1입니다.


Test Set에 대하여 Accuracy를 평가한 결과는 0.95996..이었습니다 