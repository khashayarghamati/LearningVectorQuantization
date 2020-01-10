import math
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Khashayar'
__email__ = 'khashayar@ghamati.com'


class LVQ(object):

    def __init__(self, data):
        self.data = data
        self.X = []
        self.Y = []
        self.Lables = []
        self.data_matrix = None
        self.random_lbl = []

    def get_random(self, random_number=0):

        for point in self.data:
            self.X.append(point[0])
            self.Y.append(point[1])
            self.Lables.append(point[2])


        plt.plot(self.X, self.Y, 'bo')
        plt.show()
        self.data_matrix = np.eye(2, len(self.X))
        self.data_matrix[0] = self.X
        self.data_matrix[1] = self.Y

        if random_number == 0:
            random_number /= len(self.X)/2

        random_i = np.random.choice(len(self.X), random_number)

        random_sample = self.data_matrix[:, random_i]

        for i in random_i:
            self.random_lbl.append(self.Lables[i])

        return random_sample.T

    def estimate_distance(self, random_number=0):
        random_samples = self.get_random(random_number)

        distances = []
        x = []
        y = []
        l = []

        for i, sample in enumerate(random_samples):
            for orginal in self.data_matrix.T:
                d = math.sqrt((sample[0] - orginal[0]) ** 2 + (sample[1] - orginal[1]) ** 2)
                distances.append(d)
                x.append(sample[0])
                y.append(sample[1])
                l.append(self.random_lbl[i])

        distances_m = np.eye(len(distances), 4)
        distances_m[:, 0] = x
        distances_m[:, 1] = y
        distances_m[:, 2] = distances
        distances_m[:, 3] = l

        return distances_m

    def lvq(self, random_number):
        distances = self.estimate_distance(random_number)

        distance_col = distances[:, 2]
        x_col = distances[:, 0]
        y_col = distances[:, 1]
        labels = distances[:, 3]
        alpha = .51

        new_x = []
        new_y = []

        m_in = np.argmin(distance_col)
        for i in range(random_number):
            for o in range(len(self.Lables)):
                if labels[m_in] == self.Lables[o]:
                    p1 = x_col[m_in] + alpha * (self.X[o] - x_col[m_in])
                    p2 = y_col[m_in] + alpha * (self.Y[o] - y_col[m_in])
                else:
                    p1 = x_col[m_in] - alpha * (self.X[o] - x_col[m_in])
                    p2 = y_col[m_in] - alpha * (self.Y[o] - y_col[m_in])

                new_x.append(p1)
                new_y.append(p2)

        return list(set(new_x)), list(set(new_y))


data = [
    (1, 10, -1),
    (2, 20, 1),
    (3, 30, -1),
    (4, 40, 1)
]

l = LVQ(data=data)
x, y = l.lvq(3)
plt.plot(x, y, 'ro')
plt.show()
