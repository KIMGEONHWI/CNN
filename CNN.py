# 필요한 라이브러리를 불러옵니다.
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# 이미 저장된 모델을 불러옵니다.
model = tf.keras.models.load_model('best_model.h5')

# 분류하려는 이미지를 불러옵니다. 이미지의 경로는 본인의 환경에 맞게 수정해 주세요.
img_path = '102.jpg'
img = image.load_img(img_path, target_size=(150, 150))

# 이미지를 모델이 학습할 때 사용한 것과 동일한 형태로 변환합니다.
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255. 

# 이미지를 모델에 입력하고 예측 결과를 받습니다.
preds = model.predict(img_tensor)

# 예측 결과를 Empire State Building이 있는지 없는지로 출력합니다.
# 모델이 이진 분류(binary)를 기반으로 학습되었으므로, 0.5를 기준으로 판단합니다.
if preds > 0.5199:
    print(True) # Empire State Building이 있다.
else:
    print(False) # Empire State Building이 없다.
