from keras.utils import Sequence
import numpy as np
from PIL import Image as pil

# batch만큼 데이터를 새로 메모리에 올리고 이전거는 내려버림
# 각 batch 요청때마다 getitem이 호출 되고 그 내부에서 data_generation으로 실제 데이터 읽음
# 각 epoch 끝날때마다 on_epoch_end가 호출됨, 그냥 인덱스를 섞어서 랜덤하게 batch를 뽑게함
class DataGenerator(Sequence):
    def __init__(self, trainID, targetID, trainSize, targetSize, batch_size, shuffle=True):
        self.trainID = trainID
        self.targetID = targetID
        self.trainSize = trainSize
        self.targetSize = targetSize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    # 각 epoch 마지막에 인덱스를 섞어줌
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.trainID))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, *self.trainSize, 3))
        y = np.empty((self.batch_size, *self.targetSize, 3))

        for i, ID in enumerate(list_IDs_temp):
            tmp = pil.open("E:/sourceImage/splitPictureResize/"+ID)
            x[i,] = np.asarray(tmp).reshape((self.trainSize, 3))

            tmp = pil.open("E:/sourceImage/splitPicture/"+ID)
            y[i,] = np.asarray(tmp).reshape((self.targetSize, 3))

        return x, y

    def __len__(self):
        return int(np.floor(len(self.trainID) / self.batch_size))

    def __getitem__(self, index):
        # index에 맞는 데이터 이름을 batch만큼 뽑아서 data_generation으로 전달
        # 100x100 이나 200x200 이나 이름은 같으니까 data_generation에 같은 이름 보내도 됨
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.trainID[k] for k in indexes]

        x, y = self.__data_generation(list_IDs_temp)

        return x, y

"""
# 이렇게 파일 리스트를 뽑아 위 클래스의 ID로 사용한다
# 어차피 리사이즈 된 이미지나 원본이미지나 이름은 다 같은걸로 맞춰놨다
# 이름만 뽑아서 클래스로 던져주는거니까 어디서 뽑든 상관 없다 
dir = "E:/sourceImage/splitPictureResize/"
filelist = os.listdir(dir)

print(filelist[:10])
"""