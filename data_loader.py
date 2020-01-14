
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

MAT_LIST_TRAIN = './'
MAT_LIST_TEST = './'

class MattingImageGenerator(keras.utils.Sequence):

    def __init__(self, datamode, batch_size, image_dir ,image_size=(224, 224), shuffle = True):

        self.batch_size = batch_size

        self.image_dir = image_dir

        self.shuffle = shuffle



        if datamode == "train":

            self.list_image_file = MAT_LIST_TRAIN

        elif datamode == "test":

            self.list_image_file = MAT_LIST_TEST

        else:

            raise Exception('Setting a wrong data mode, should be either training. validation or test')

        self.datamode = datamode

        self.image_size = image_size



        self.list_paths = None

        self.__load_list_path(self.list_image_file)



        self.indexes = np.arange(len(self.list_paths))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __len__(self):

        return int(math.floor(len(self.list_paths) / float(self.batch_size)))



    def __load_list_path(self, list_image_file):

        with open(list_image_file) as file:

            self.list_paths = file.read().splitlines()



    def __getitem__(self, index):

        local_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        x = np.empty([self.batch_size] + list(self.image_size) + [3], dtype=np.float32)

        y = np.empty([self.batch_size] + list(self.image_size) + [3], dtype=np.float32)



        for j in range(self.batch_size):

            image_path = self.list_paths[local_indexes[j]]

            matting_path = self.list_paths[local_indexes[j]]

            image = image_preprocess(cv2.imread(image_path))


            x[j, ...] = image

            y[j, ...] = image_preprocess(cv2.imread(matting_path))


        return x, y

