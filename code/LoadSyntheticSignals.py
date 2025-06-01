# Generator synthetic Running data 
# Made them into a PyTorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
from scipy import io  # برای بارگذاری داده‌های واقعی از فایل‌های MATLAB
import pandas as pd
import matplotlib.pyplot as plt

class Synthetic_Dataset(Dataset):
    def __init__(self, 
                 Running_model_path='./logs/3_STORY_2025_04_15_15_15_00/Model/checkpoint',
                 adl_data_path='./LosAlamos_Baseline_L1024_t0031/data/adl_data.mat',
                 sample_size=1):
        
        self.sample_size = sample_size
        

        adl_data = io.loadmat(adl_data_path)['adl_data']

        num_samples = 1025
        adl_data = np.reshape(adl_data, (-1, num_samples, 3), order='F')
        

        self.data_min = np.min(adl_data)
        self.data_max = np.max(adl_data)
        print(f"Min value of real data: {self.data_min}, Max value of real data: {self.data_max}")
        
        #  (Generator)
        running_gen_net = Generator(seq_len=1024, channels=3, latent_dim=256)
        running_ckp = torch.load(Running_model_path)
        running_gen_net.load_state_dict(running_ckp['gen_state_dict'])
        

        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 256)))
        self.syn_running = running_gen_net(z)
        self.syn_running = self.syn_running.detach().numpy()
        

        self.syn_running = self.syn_running * (self.data_max - self.data_min) + self.data_min
        #self.syn_running[:, 2, :, :] = self.syn_running[:, 2, :, :] + 6

        self.running_label = np.zeros(len(self.syn_running))
        
        self.combined_train_data = self.syn_running
        self.combined_train_label = self.running_label
        
        print(self.syn_running.shape)
        print(self.running_label.shape)
        
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        return self.combined_train_data[idx], self.combined_train_label[idx]

if __name__ == "__main__":

    syn_dataset = Synthetic_Dataset()


    df = pd.DataFrame(np.transpose(syn_dataset[0][0][:, 0, :]))
    df.to_csv("denormalized_data.csv", header=False, index=False)
    print("Data has been saved to 'denormalized_data.csv'")
 

    syn_signals = syn_dataset[0][0]
    time_interval = 0.0031
    time_vector = np.arange(0, syn_signals.shape[-1] * time_interval, time_interval)
    print("Shape of syn_signals:", syn_signals.shape)
    

    print("Shape of syn_signals:", syn_signals.shape)


    time_interval = 0.0031
    time_vector = np.arange(0, syn_signals.shape[-1] * time_interval, time_interval)

    plt.figure(figsize=(15, 5))


    plt.plot(time_vector, syn_signals[0, 0, :], label='X')
    plt.plot(time_vector, syn_signals[1, 0, :], label='Y')
    plt.plot(time_vector, syn_signals[2, 0, :], label='Z')


    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("output.png")
    from IPython.display import Image
    Image("output.png")
