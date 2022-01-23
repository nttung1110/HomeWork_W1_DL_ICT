import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb


class TwoDimRecSamplingData(torch.utils.data.Dataset):
    def __init__(self, num_data, mode, uni_range=[-5, 5]):
        self.num_data = num_data
        self.uni_range = uni_range
        self.mode = mode
        self.radius_boundary_circle = (self.uni_range[1] - self.uni_range[0])/2
        self.center_circle_range = [(self.uni_range[1] + self.uni_range[0])/2,
                                    (self.uni_range[1] + self.uni_range[0])/2]

        # generate data list
        self.data, self.label, self.color_vis = self.uniform_sampling(self.num_data)
        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        # construct label for the data
        return torch.FloatTensor(self.data[index]), self.label[index]
            
    def uniform_sampling(self, num_point):
        # data_x = np.random.uniform(low=self.uni_range[0], high=self.uni_range[1], size=(num_point, 1))
        # data_y = np.random.uniform(low=self.uni_range[0], high=self.uni_range[1], size=(num_point, 1))
        # data = np.concatenate((data_x, data_y), axis=1)
        
        data = np.random.uniform(low=[self.uni_range[0], self.uni_range[0]], 
                                high=[self.uni_range[1], self.uni_range[1]],
                                size=(num_point, 2))

        label = np.empty((data.shape[0])) 
        color_vis = []
        for i in range(data.shape[0]):
            [x_point, y_point] = data[i]

            # check label
            length_x = (x_point-self.center_circle_range[0])**2
            length_y = (y_point-self.center_circle_range[1])**2

            length = (length_x + length_y) ** (1/2)

            if length < self.radius_boundary_circle:
                label[i] = 1
                color_vis.append('blue')
            else:
                label[i] = 0
                color_vis.append('red')
            

        return data, label, color_vis

    def visualize(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        x = self.data[:, 0]
        y = self.data[:, 1]
        ax.scatter(x, y, c=self.color_vis)
        plt.savefig("img_vis_data_"+self.mode+".png")


if __name__ == "__main__":
    # for testing only
    my_dataset = TwoDimRecSamplingData(10000, "train")
    my_dataset.visualize()

    for idx_data in range(len(my_dataset)):
        item = my_dataset[idx_data]
        pdb.set_trace()