import numpy as np
import os
import framework.config as config


def collect_label(training_file):
    label_set = set()
    with open(training_file, 'r') as f:
        for line in f.readlines()[1:]:
            part = line.split('\n')[0].split(',')
            [ParticipantID, Substrate, Cover, Gesture, Sequence] = part[:5]
            label_set.add(Gesture)
    print(label_set)
    print(len(label_set))



class DataGenerator_data(object):
    def __init__(self, batch_size, normalization, window=None, seed=1234):

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        # Load data
        datapath = os.path.join(os.getcwd(), 'dataset')

        if window is not None:
            x_file = os.path.join(datapath, '1_haart_train_x' + str(window) + '.txt')
            y_file = os.path.join(datapath, '1_haart_train_y' + str(window) + '.txt')
            val_x_file = os.path.join(datapath, '1_haart_val_x' + str(window) + '.txt')
            val_y_file = os.path.join(datapath, '1_haart_val_y' + str(window) + '.txt')
        else:
            x_file = os.path.join(datapath, '1_haart_train_x.txt')
            y_file = os.path.join(datapath, '1_haart_train_y.txt')
            val_x_file = os.path.join(datapath, '1_haart_val_x.txt')
            val_y_file = os.path.join(datapath, '1_haart_val_y.txt')

        if os.path.exists(x_file):
            self.train_x, self.train_y = np.loadtxt(x_file), np.loadtxt(y_file)
            self.val_x, self.val_y = np.loadtxt(val_x_file), np.loadtxt(val_y_file)
        else:
            validation_file = os.path.join(datapath, 'testWITHLABELS.csv')
            training_file = os.path.join(datapath, 'training.csv')

            self.train_x, self.train_y = self.read_x_y(training_file, datatype='training')
            # print(self.train_x.shape, self.train_y.shape)
            self.train_x, self.train_y = self.reshape_moduel(self.train_x, self.train_y, window)

            self.val_x, self.val_y = self.read_x_y(validation_file, datatype='val')
            self.val_x, self.val_y = self.reshape_moduel(self.val_x, self.val_y, window)

            np.save(x_file, self.train_x)
            np.save(y_file, self.train_y)
            np.save(val_x_file, self.val_x)
            np.save(val_y_file, self.val_y)

        print('training: ', self.train_x.shape,  self.train_y.shape)
        print('testing: ', self.val_x.shape, self.val_y.shape)
        # (578, 432, 8, 8) (578,)
        # (251, 432, 8, 8) (251,)

        # self.normal = normalization
        # if self.normal:
        #     # print(x.shape)
        #     x = x.reshape(x.shape[0], x.shape[1], -1)
        #     # print(x.shape)
        #     # (578, 432, 64)
        #     (self.mean, self.std) = calculate_scalar(self.train_x)


    def reshape_moduel(self, train_x, train_y, window):
        ori_shape = train_x.shape
        train_x = train_x.reshape(ori_shape[0], -1, window, ori_shape[-2], ori_shape[-1])
        train_y = train_y[:, None]
        train_y = np.tile(train_y, (1, int(ori_shape[1] / window)))
        # print(self.train_x.shape, self.train_y.shape)
        # # (578, 8, 54, 8, 8) (578, 8)
        # print(self.train_y[:10])
        # # [[5 5 5 5 5 5 5 5]
        # #  [0 0 0 0 0 0 0 0]
        # #  [4 4 4 4 4 4 4 4]
        # #  [2 2 2 2 2 2 2 2]
        # #  [1 1 1 1 1 1 1 1]
        # #  [6 6 6 6 6 6 6 6]
        # #  [3 3 3 3 3 3 3 3]
        # #  [5 5 5 5 5 5 5 5]
        # #  [0 0 0 0 0 0 0 0]
        # #  [4 4 4 4 4 4 4 4]]
        train_x = train_x.reshape(-1, window, ori_shape[-2], ori_shape[-1])
        train_y = train_y.flatten()
        # print(self.train_x.shape, self.train_y.shape)
        # (4624, 54, 8, 8) (4624,)
        return train_x, train_y


    def read_x_y(self, file, datatype):
        feature = []
        all_label_y = []

        each_ParticipantID, each_Substrate, each_Cover, each_Gesture = [], [], [], []
        each_matrix = []
        each_label = []
        with open(file, 'r') as f:
            for line in f.readlines()[1:]:
                part = line.split('\n')[0].split(',')

                if datatype=='training':
                    [ParticipantID, Substrate, Cover, Gesture] = part[:4]
                    channels = [float(each) for each in part[4:]]
                else:
                    [ParticipantID, Substrate, Cover, Gesture, Sequence] = part[:5]
                    # "ParticipantNo", "Substrate", "Cover", "Gesture",
                    channels = [float(each) for each in part[5:]]

                each_ParticipantID.append(ParticipantID)
                each_Substrate.append(Substrate)
                each_Cover.append(Cover)
                each_Gesture.append(Gesture)

                if len(channels) != 64:
                    print(file)
                    print(line)
                    print(channels)
                    print(len(channels))
                channel_matrix = np.array(channels).reshape(8, 8)
                # print(subject, variant, gesture)
                # print(channel_matrix)
                each_matrix.append(channel_matrix)

                label_id = config.lb_to_ix[Gesture]
                # print(Gesture, label_id)
                each_label.append(label_id)

                if len(each_ParticipantID) == 432:
                    assert len(set(each_ParticipantID)) == 1
                    assert len(set(each_Substrate)) == 1
                    assert len(set(each_Cover)) == 1
                    assert len(set(each_Gesture)) == 1
                    assert len(set(each_label)) == 1

                    feature.append(np.array(each_matrix))
                    all_label_y.append(each_label[0])

                    each_ParticipantID, each_Substrate, each_Cover, each_Gesture = [], [], [], []
                    each_matrix = []
                    each_label = []

        # print(np.array(feature).shape, np.array(all_label_y).shape)
        return np.array(feature), np.array(all_label_y)


    def generate_train(self):
        """Generate mini-batch data for training.

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = list(range(len(self.train_y)))
        # print(audio_indexes)

        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.train_x[batch_audio_indexes]
            # if self.normal:
            #     # Transform data
            #     batch_x = self.transform(batch_x)

            batch_y = self.train_y[batch_audio_indexes]

            # print(batch_x.shape, batch_y.shape)

            yield batch_x, batch_y


    def generate_validate(self, data_type, shuffle, devices=['a'], max_iteration=None):
        """Generate mini-batch data for evaluation.

        Args:
          data_type: 'train' | 'validate'
          devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
          max_iteration: int, maximum iteration for validation
          shuffle: bool

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        batch_size = self.batch_size

        audio_indexes = list(range(len(self.val_y)))

        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} samples in validation'.format(
             len(audio_indexes)))

        audios_num = len(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            # print(batch_audio_indexes)
            pointer += batch_size

            iteration += 1

            batch_x = self.val_x[batch_audio_indexes]
            # if self.normal:
            #     # Transform data
            #     batch_x = self.transform(batch_x)

            batch_y = self.val_y[batch_audio_indexes]

            # print(batch_y)
            yield batch_x, batch_y


    # def transform(self, x):
    #     """Transform data.
    #
    #     Args:
    #       x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)
    #
    #     Returns:
    #       Transformed data.
    #     """
    #
    #     return scale(x, self.mean, self.std)


