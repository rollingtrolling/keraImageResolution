# 20200915_resizeGUI 에서 만들어진 모델을 불러와 이미지를 복원
# 학습은 쪼개진 이미지를 입력받아 했지만 복원은 합쳐진 이미지를 입력받아 한다
# 이미지 입력 > 100x100 크기로 쪼갬 > 각각 모델에 넣어 복원 > 다시 합침
# 겉 포장은 pyqt를 이용해 윈도우 gui로 만든다
# 학습된 모델이 색이 변색되는 문제가 있지만 일단 넘어감, 합친 이미지로 비교했을 때 문제가 심각하면 수정

# 해야할 것
# 1. 이미지를 합쳐 출력하는 부분 코딩
# 2. 합친 이미지의 패딩 부분 삭제
# 3. 지금 모델은 원본이미지 > 축소 > 다시 확대 하는 구조이므로 축소하는 과정을 없애고 다시 학습 & 모델 저장
# 4. 만약 모델 정확도를 올리고 싶으면 학습 이미지 량을 늘린다(지금은 2000개)
# 4. pyqt로 gui - 이미지 입력

import numpy as np
from keras.models import load_model
from PIL import Image as pil
import matplotlib.pyplot as plt

model = load_model('zoomModel.h5')


splitPicDir = "E:\sourceImage\splitPicture\\"
splitResizeDir = "E:\sourceImage\splitPictureResize\\"


imageNum = 0
splitImageNum = 20 # 0번 이미지가 20개로 나뉘어졌으므로, 전체 이미지 적용시 변수로 수정

# 이미지 넘파이리스트를 받아 병합
def imageMerge(list):
    # 테스트 이미지가 가로 4개 세로 5개로 쪼개짐
    widthNum = 4
    heightNum = 5

    canvas = pil.new('RGB', (200*widthNum, 200*heightNum))
    for widthIdx in range(widthNum):
        for heightIdx in range(heightNum):
            tmp = pil.fromarray(list[widthIdx*5+heightIdx].astype(np.uint8))
            canvas.paste(tmp, (widthIdx*200, heightIdx*200))

    return canvas


# 패딩된 이미지와 원래크기를 입력받아 패딩 부분을 제거
def deletePadding(image, sizeX, sizeY):
    area = (0, 0, sizeX, sizeY)
    cropped = image.crop(area)

    return cropped


# ------------------------------------------------------------------------------------------------
# imageMerge Test
imageList = np.empty((0, 200, 200, 3))
for i in range(20):
    tmp = pil.open(splitPicDir+"img-00000-"+str(i).zfill(5)+".jpg")
    tmp = np.asarray(tmp)
    tmp = tmp.reshape((1, 200, 200, 3))
    #tmp = np.array(tmp).reshape((1, 200, 200, 3))
    #tmp = tmp / 255
    imageList = np.concatenate((imageList, tmp), axis=0)

mergedImage = imageMerge(imageList)

plt.imshow(mergedImage)
plt.show()

#test = pil.open(splitPicDir+"img-00000-"+str(6).zfill(5)+".jpg")
deleted = deletePadding(mergedImage, 722, 999)

plt.imshow(deleted)
plt.show()

deleted.save("test1.jpg")


# ------------------------------------------------------------------------------------------------
# original
"""
imageList = np.empty((0, 200, 200, 3))
for splitNum in range(splitImageNum):
    splitImage = pil.open(splitPicDir+"img-"+str(imageNum).zfill(5)+"-"+str(splitNum).zfill(5)+".jpg")
    splitImage = np.array(splitImage).reshape((1,200,200,3))
    splitImage = splitImage.astype('float32')
    splitImage = splitImage / 255

    predictedImage = model.predict(splitImage)
    predictedImage = predictedImage.reshape(200, 200, 3)

    imageList = np.concatenate((imageList, predictedImage), axis=0)

    # 모델 통과 후 이미지들을 모아서 배열로 만듦
    # 배열을 이미지병합하는 함수로 넘겨줌

mergedImage = imageMerge(imageList)

plt.imshow(mergedImage)
plt.show()
"""