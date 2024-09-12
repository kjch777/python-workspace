import pickle
import tensorflow as tf
import numpy as np

########## 모델 로드

# Keras 모델 로드
model = tf.keras.models.load_model('./samsung_electronics_stock_close_price_time_series_regression_model/model.h5')

with open('./transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

########## 모델 예측

# Prepare the input data as a column vector for correct scaling
x_test = np.array([[30000.000000], 
                   [29300.000000], 
                   [30000.000000], 
                   [29980.000000], 
                   [29700.000000], 
                   [29020.000000], 
                   [28740.000000]])

# Transform each value separately and then reshape the array back
x_test = transformer.transform(x_test).reshape(1, -1, 1)
print(x_test)

# 예측 수행
y_predict = model.predict(x_test)
print("예상하는 주식 가격")
print(y_predict)  # 예측된 결과

# 역변환하여 실제 값으로 변환
inverse = transformer.inverse_transform([[y_predict.flatten()[0]]])
print(inverse.flatten()[0])