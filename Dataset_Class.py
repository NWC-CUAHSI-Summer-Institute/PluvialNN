#!/usr/bin/env python
# coding: utf-8

# ### Note:
# 
# This notebook is for creating the dataset for NN model.

# In[7]:


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from torchvision import transforms


# In[39]:


class Pluvial_Dataset(Dataset):
    def __init__(self, rainfall_path, flood_folder, building_path, curve_path, DEM_path, time_of_concentration, threshold = 0.3):
        # Load rainfall data
        df = pd.read_excel(rainfall_path, header=0)
        rainfall_series = df.drop(columns=["Time"]).astype(float).reset_index(drop=True)
        self.rainfall = [torch.tensor(rainfall_series.iloc[:, i].values.astype(np.float32)).unsqueeze(1)
                         for i in range(rainfall_series.shape[1])]
        
        # Calculate rainfall durations
        self.durations = []
        for r in self.rainfall:
            r_np = r.squeeze().numpy()
            non_zero = np.nonzero(r_np)[0]
            duration = 0 if len(non_zero) == 0 else (non_zero[-1] - non_zero[0] + 1)
            self.durations.append(torch.tensor([duration], dtype=torch.float32))
        
        # Load building, curve number, and DEM rasters (shared for all events)
        with rasterio.open(building_path) as src:
            self.buildings = torch.tensor(src.read(1).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        with rasterio.open(curve_path) as src:
            self.curve = torch.tensor(src.read(1).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        with rasterio.open(DEM_path) as src:
            self.DEM = torch.tensor(src.read(1).astype(np.float32), dtype=torch.float32).unsqueeze(0)

        self.flood_folder = flood_folder
        self.time_of_concentration = torch.tensor([time_of_concentration], dtype=torch.float32)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.threshold = threshold
    
    def __len__(self):
        return len(self.rainfall)
    
    def __getitem__(self, idx):
        # Rainfall sequence
        rain = self.rainfall[idx]
        dur = self.durations[idx]
        tc = self.time_of_concentration
        curve = self.curve
        DEM = self.DEM
        build = self.buildings

        # Load corresponding flood raster
        event_folder = os.path.join(self.flood_folder, f"{idx}")
        flood_path = os.path.join(event_folder, f"{idx}DepthFLT", f"{idx}Depth.flt")
        with rasterio.open(flood_path) as src:
            flood = torch.tensor(src.read(1).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        flood[flood > 5] = 0
        flood = (flood > self.threshold).to(torch.uint8)

        return {
            "rainfall": rain,
            "duration": dur,
            "tc": tc,
            "curve": curve,
            "DEM": DEM,
            "buildings": build,
            "flood": flood
        }

