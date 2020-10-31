# 계획
#   윈도우 gui용 해상도 상승 어플 개발
# 기능
#   어플 내 러닝 버튼 클릭시 지정된 파일 내부에 있는 이미지들로 학습
#   어플로 이미지를 넣고 예측 버튼 클릭시 러닝된 값으로 이미지 해상도가 증가된 값 출력

#import keras
# from keras.datasets import mnist
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
        #self.imgWidth = 100
        #self.imgHeight = 100
        self.resizeSize = 100
        self.originalSize = 200

        self.splitPicDir = "E:\sourceImage\splitPicture\\"
        self.splitResizeDir = "E:\sourceImage\splitPictureResize\\"

        self.zoomModel = self.buildModel()
        #self.zoomModel.compile(optimizer='adam', loss='binary_crossentropy')

        self.generator = DataGenerator()


    def buildModel(self):
        model = Sequential()

        """
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(self.originalSize, self.originalSize, 3)))
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))

        model.add(Conv2DTranspose(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(self.resizeSize, self.resizeSize, 3)))
        model.add(Conv2DTranspose(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2DTranspose(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))

        model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))
        """
        model.add(Conv2DTranspose(2, kernel_size=(25, 25), activation='relu', kernel_initializer='he_normal', input_shape=(100, 100, 3)))
        model.add(Conv2DTranspose(4, kernel_size=(25, 25), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2DTranspose(8, kernel_size=(25, 25), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2DTranspose(16, kernel_size=(25, 25), activation='relu', kernel_initializer='he_normal'))
        model.add(Conv2DTranspose(32, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal'))

        model.add(Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same'))

        model.summary()

        return model


    def imageRead(self, url, imageSize):
        loadedImage = np.empty((0, imageSize, imageSize, 3))
        i = 0
        present = time.time()
        for dirpath, dirnames, filenames in os.walk(url):
            for filename in filenames:
                tempImage = pil.open(url+filename)
                tempImage = np.array(tempImage).reshape((1, imageSize, imageSize, 3))
                loadedImage = np.concatenate((loadedImage, tempImage), axis=0)

                i = i+1
                if i % 1000 == 0:
                    print("loop : ", i, ", time takes ", time.time()- present, " sec")
                    present = time.time()
                    print (loadedImage.shape)
                if i == 2000:
                    break

        loadedImage = np.array(loadedImage.astype('float32'))
        loadedImage = loadedImage / 255

        # (n, 200, 200, 3) 모양의 넘파이 배열 loadedImage가 리턴 됨
        # 이게 train 메서드의 input_train 변수로 들어감
        # train에서 이미지를 반으로 줄인 후 transpose convolution으로 다시 확대해서 학습
        return loadedImage


    def train(self, epochs, batchSize, validationSplit):
        """
        # load dataset and normalize
        (input_train, target_train), (input_test, target_test) = mnist.load_data()

        input_train = input_train.reshape(input_train.shape[0], self.imgWidth, self.imgHeight, 1)
        input_test = input_test.reshape(input_test.shape[0], self.imgWidth, self.imgHeight, 1)

        input_train = np.array(input_train.astype('float32'))
        input_test = np.array(input_test.astype('float32'))

        input_train = input_train / 255
        input_test = input_test / 255
        """
        input_train = self.imageRead(self.splitResizeDir, self.resizeSize)
        target_test = self.imageRead(self.splitPicDir, self.originalSize)


        num_reconstructions = 3
        self.samples = input_train[:num_reconstructions]
        self.targets = target_test[:num_reconstructions]

        self.zoomModel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("fitting start")
        start = time.time()
        #result = self.zoomModel.fit(input_train, target_test, epochs=epochs, batch_size=batchSize, validation_split=validationSplit)
        result = self.zoomModel.fit_generator(generator=input_train, epochs=epochs)
        print("fitting end, time takes ", time.time()-start, " sec")

        return result


    def visualize(self,):
        num_reconstructions = 3
        reconstructions = self.zoomModel.predict(self.samples)
        print(reconstructions.shape)

        for i in np.arange(0, num_reconstructions):
            sample = self.samples[i]
            original = self.targets[i]
            reconstruction = reconstructions[i]

            #input_class = self.targets[i]

            fig, axes = plt.subplots(1, 2)

            axes[0].imshow(original)
            axes[0].set_title('original image')
            axes[1].imshow(reconstruction)
            axes[1].set_title('reconstrucion with con2dtranspose R')

            #fig.suptitle(f'mnist target = {input_class}')
            plt.show()


    def modelPredict(self, image):
        # image shape = (1, 200, 200, 3) > (1, 100, 100, 3)
        predictedImage = self.zoomModel.predict(image)
        predictedImage = predictedImage.reshape(200, 200, 3)

        plt.imshow(predictedImage)
        plt.title('predicted image')

        plt.show()


    def modelSave(self):
        #self.zoomModel.save('zoomModel.h5')
        # 이미지 축소>확대 모델에서 이미지 확대 모델로 변환 후
        self.zoomModel.save('zoomModel2.h5')


if __name__ == '__main__':
    zoom = Zoom()
    result = zoom.train(100, 10, 0.1)
    zoom.visualize()

    plt.plot(result.history['acc'])
    plt.plot(result.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper_left')
    plt.show()

    plt.plot(result.history['loss'])
    plt.plot(result.histroy['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """
    zoom = Zoom()
    zoom.train(10, 10, 0.2)
    zoom.visualize()

    testImage = pil.open("E:\sourceImage\splitPictureResize\img-06954-00017.jpg")
    testImage = np.array(testImage).reshape((1, 100, 100, 3))
    testImage = np.array(testImage).astype('float32')
    testImage = testImage / 255
    zoom.modelPredict(testImage)

    zoom.modelSave()
    """

# 현재 문제 2가지
# - 메모리 부족으로 데이터를 최대 2000개 밖에 못 넣음 > 가진 데이터는 몇십만갠데 아까움
# - 학습 loss 값이 잘 안줄음 > 데이터를 많이 넣거나 compile 값 조정 필요
# 해결책
# - fit_generator나 hdf5 사용으로 넣는 데이터 최적화
# - loss 값 조정은 데이터를 떄려넣은 후에 다시 확인하자

# 테스트 때는 jpg를 읽는 것보다 npy를 읽는게 훨씬 빠름 > 사전에 jpg를 npy로 변환 후 읽자
# 용량 때문에 3000개만 변환 완료됨, npy 읽는 구조로 바꾸고 시간 비교
# > 실제 읽는 시간은 별로 차이 없더라 아마 시간이 걸리는 건 읽을 때가 아니라 읽고 배열로 concatenate할때 인듯
# > 따라서 fit_generator로 그때그떄 읽어서 학습하면 concatenate 시간이 줄어드니 npy로 변환 안해도 시간 단축 될 것

# fit_generator
# https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# http://www.kwangsiklee.com/2018/11/keras%EC%97%90%EC%84%9C-sequence%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8C%80%EC%9A%A9%EB%9F%89-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B-%EC%B2%98%EB%A6%AC%ED%95%98%EA%B8%B0/
# https://leestation.tistory.com/776