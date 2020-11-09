import numpy as np

# 1.  데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
# y = ax + b

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델 구성
model = Sequential() # 순차적이다.
model.add(Dense(5, input_dim=1)) # 한개가 입력됬다
model.add(Dense(5)) # 다음층에 5개의 요오드가 생성
model.add(Dense(5))
model.add(Dense(1)) # 결과 weight
# 요오드의 갯수와 레이어의 깊이는 AI Developer가 정한다.
# * Hyper Parameter Tuning *

# 1 - 3 - 5 - 3 - 1 의 구조
# Dense = DNN * 추가로 알아봐야될듯

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['accuracy'])

    # loss(mse) = 손실함수는 정답에 대한 오류를 숫자로 나타내는 것으로 오답에 가까울수록 큰 값이 나온다. 반대로 정답에 가까울수록 작은 값이 나온다.
    # Optimizer(최적화) = adam
    # Metrics = 평가 지표.

model.fit(x, y, epochs=3000, batch_size=1)
    # epochs = 훈련 횟수
    # batch_size = 몇 개의 샘플로 가중치를 갱신할 것인지 지정.

# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss : ", loss)
print("acc : ", acc)





