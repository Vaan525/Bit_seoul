# 실습
# R2를 음수가 아닌 0.5 이하로 줄이자
# 레이어는 인풋과 아웃풋을 포함한 7개 이상(히든이 5개 이상) .
# 히든레이어 노드는 레이어당 각각 최소 10개 이상
# batch_size = 1
# epochs = 100 이상
# 데이터 조작 금지

import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 테스트 하고싶은 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) # 평가 하고싶은 데이터
y_test = np.array([11,12,13,14,15]) 
x_pred = np.array([16,17,18]) # 예측하고 싶은 데이터


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(50))
model.add(Dense(200))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))
# 모델구성단계에서 번잡하게 만든다. 그럼 0.5 이하로

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
# print("acc : ", acc)

# 4. 예측
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)
# 새로운 예측값을 만들어서 아래에서 비교

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
# R2 값은 회귀 모델에서 예측의 적합도를 0과 1사이의 값으로 계산
# 1에 가까울수록 완벽


