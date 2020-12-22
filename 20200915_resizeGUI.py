from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
import matplotlib.pyplot as plt
import numpy as np

import os
from PIL import Image as pil
import time

from kerasDataGenerator import DataGenerator


class Zoom():
    def __init__(self):
        self.resizeSize = 100
        self.originalSize = 200

        self.splitPicDir = "E:\sourceImage\splitPicture\\"
        self.splitResizeDir = "E:\sourceImage\splitPictureResize\\"

        self.zoomModel = self.buildModel()


    def buildModel(self):
        model = Sequential()

        # 200x200으로 확대된 이미지를 convolution > transposeConvolution으로 보간 작업만 하는 모델
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(200, 200, 3)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))

        model.add(Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(194, 194, 3)))
        model.add(Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        
        model.add(Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same'))

        model.summary()

        return model


    def train(self, epochs, batchSize, stepsPerEpoch):
        trainIDs = os.listdir(self.splitResizeDir)#***경로 수정
        customGen = DataGenerator(trainIDs, batchSize, self.originalSize)

        #self.zoomModel.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
        #self.zoomModel.compile(optimizer='Adam', loss='mean_absolute_error', metrics=['accuracy'])
        self.zoomModel.compile(optimizer='Adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

        print("fitting start")
        start = time.time()
        result = self.zoomModel.fit_generator(generator=customGen,
                                              epochs=epochs,
                                              steps_per_epoch=stepsPerEpoch)
        print("fitting end, time takes ", time.time()-start, " sec")

        return result


    def visualize(self, testImageNum, testSplitNum):
        testImage = pil.open(self.splitResizeDir+'img-'+str(testImageNum).zfill(5)+'-'+str(testSplitNum).zfill(5)+'.jpg')#*** 경로 수정
        testImage = testImage.resize((self.originalSize, self.originalSize))
        testImage = np.asarray(testImage).reshape((1, self.originalSize, self.originalSize, 3))

        reconstruction = self.zoomModel.predict(testImage / 255)
        reconstruction = reconstruction * 255
        original = pil.open(self.splitPicDir+'img-'+str(testImageNum).zfill(5)+'-'+str(testSplitNum).zfill(5)+'.jpg')#*** 경로 수정

        testImage = testImage.reshape((self.originalSize, self.originalSize, 3))
        reconstruction = reconstruction.reshape((self.originalSize, self.originalSize, 3))

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(testImage.astype(np.uint8))
        axes[0].set_title('input image')
        axes[1].imshow(reconstruction.astype(np.uint8))
        axes[1].set_title('reconstructed image')
        axes[2].imshow(original)
        axes[2].set_title('original image')

        plt.show()


    # 이미지(numpy array)를 학습된 모델에 넣어 결과값(numpy array) 리턴
    def modelPredict(self, image):
        predictedImage = self.zoomModel.predict(image)
        predictedImage = predictedImage.reshape(200, 200, 3)

        return predictedImage


    def modelSave(self):
        self.zoomModel.save('zoomModel.h5')


if __name__ == '__main__':
    zoom = Zoom()
    result = zoom.train(epochs=500, batchSize=10, stepsPerEpoch=10)

    zoom.modelSave()
    zoom.visualize(0, 11)

    """
    # 모델 학습 확인용이므로 학습이 잘 되면 주석부분 삭제+visualize method 주석화
    # predict method만 남겨둔다
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(result.history['loss'], 'y', label='train loss')
    acc_ax.plot(result.history['accuracy'], 'b', label='train acc')

    loss_ax.set_xlabel('epoch')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()
    """
