# model load and test
import numpy as np
from keras.models import load_model
from PIL import Image as pil
import matplotlib.pyplot as plt


splitPicDir = "E:\sourceImage\splitPictureVer2\\"
splitResizeDir = "E:\sourceImage\splitPictureResizeVer2\\"


def imagePadding(img):
    imgWidth = img.size[0]
    imgHeight = img.size[1]

    widSpare = imgWidth % 90
    heiSpare = imgHeight % 90

    widNum = int(np.ceil(imgWidth / 90)) - 1
    heiNum = int(np.ceil(imgHeight / 90)) - 1

    if imgWidth % 90 > 5 or imgWidth % 90 == 0:
        widNum = widNum + 1
        widSpare = 0
    if imgHeight % 90 > 5 or imgHeight % 90 == 0:
        heiNum = heiNum + 1
        heiSpare = 0

    canvas = pil.new("RGB", (widNum*90+10, heiNum*90+10), (255,255,255))
    canvas.paste(img, (5,5))

    return canvas, (widSpare, heiSpare)


# pillow 이미지를 받아 pillow 리스트로 스플릿
def imageSplit(img):
    imgArray = []

    imgWidth = img.size[0]
    imgHeight = img.size[1]

    for wid in range(int(np.floor(imgWidth)/90)):
        for hei in range(int(np.floor(imgHeight)/90)):
            imgArray.append(img.crop((wid*90, hei*90, wid*90+90+2*5, hei*90+90+2*5)))

    return imgArray


# 이미지 넘파이리스트를 받아 pillow 이미지로 병합
def imageMerge(list, width, height, spare):
    # 테스트 이미지가 가로 4개 세로 5개로 쪼개짐
    widthNum = int((width - 20) / 180)
    heightNum = int((height - 20) / 180)

    canvas = pil.new('RGB', (180*widthNum+spare[0]*2, 180*heightNum+spare[1]*2))
    for widthIdx in range(widthNum):
        for heightIdx in range(heightNum):
            widSp = heiSp = 0
            if widthIdx == (widthNum-1):
                widSp = spare[0]*2
            if heightIdx == (heightNum-1):
                heiSp = spare[1]*2

            tmp = pil.fromarray(list[widthIdx*heightNum+heightIdx].astype(np.uint8), 'RGB')
            tmp = tmp.crop((10, 10, 190+widSp, 190+heiSp))
            canvas.paste(tmp, (widthIdx*180, heightIdx*180))

    return canvas


# 패딩된 이미지(pillow)와 원래크기(int)를 입력받아 패딩 부분을 제거 후 이미지(pillow) 리턴
def deletePadding(image, sizeX, sizeY):
    area = (0, 0, sizeX, sizeY)
    cropped = image.crop(area)

    return cropped


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = load_model('zoomModel.h5')

    filename = 'img-00355.jpg'
    testImage = pil.open('E:/sourceImage/pictureResize/'+filename)
    #targetImage = pil.open('e:/sourceImage/picture/'+filename)
    imgWidth = testImage.size[0]
    imgHeight = testImage.size[1]

    padded, spare = imagePadding(testImage)
    splitedImage = imageSplit(padded)

    predicted = np.zeros((len(splitedImage), 200, 200, 3))
    for i in range(len(splitedImage)):
        piece = splitedImage[i].resize((200, 200))
        piece = np.array(piece).reshape((1, 200, 200, 3))
        piece = model.predict(piece / 255) * 255
        predicted[i] = piece

    merged = imageMerge(predicted, padded.size[0]*2, padded.size[1]*2, spare)
    unpadded = deletePadding(merged, testImage.size[0]*2, testImage.size[1]*2)

    unpadded.save('E:/sourceImage/test.jpg')
