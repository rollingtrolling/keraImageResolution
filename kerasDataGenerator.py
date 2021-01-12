from keras.utils import Sequence
import numpy as np
from PIL import Image as pil


class DataGenerator(Sequence):
    def __init__(self, trainURL, targetURL, ID, batch_size, imgSize, shuffle=True):
        self.trainURL = trainURL
        self.targetURL = targetURL
        self.ID = ID
        self.imgSize = imgSize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    # 각 epoch 마지막에 인덱스를 섞어줌
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ID))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # train 이미지는 200x200으로 변환 후 리턴, target 이미지는 200x200 그대로 리턴
    def __data_generation__(self, list_IDs_temp):
        x = np.empty((self.batch_size, self.imgSize, self.imgSize, 3))
        y = np.empty((self.batch_size, self.imgSize, self.imgSize, 3))

        for i, ID in enumerate(list_IDs_temp):
            train = pil.open(self.trainURL + ID)
            train = train.resize((self.imgSize, self.imgSize))
            x[i,] = np.asarray(train).reshape((self.imgSize, self.imgSize, 3))

            target = pil.open(self.targetURL + ID)
            y[i,] = np.asarray(target).reshape((self.imgSize, self.imgSize, 3))

        return x/255, y/255


    # 한 epoch에 존재하는 batch 수 return
    def __len__(self):
        return int(np.floor(len(self.ID) / self.batch_size))


    def __getitem__(self, index):
        # index에 맞는 데이터 이름을 batch만큼 뽑아서 data_generation으로 전달
        # 100x100과 200x200이 서로 이름은 같으므로 data_generation에 같은 이름 보내도 됨
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.ID[k] for k in indexes]

        x, y = self.__data_generation__(list_IDs_temp)

        return x, y
