import sys
directory = '/home/yash/Documents/nerf_simulation/'
sys.path.insert(0, directory)

import matplotlib.pyplot as plt
import time
import torch
import numpy as np
from omegaconf import OmegaConf
import commentjson as json
from model_tcnn import NeRF_Network, object_Network
from torch.utils.data import Dataset, DataLoader
from render_ray import render_image
from sampling import NeRF_sampling

from data import scene_data
from data_loader import scene_dataset
from run_NeRF import NeRF_pipeline
import socket
from pickle import loads, dumps

from get_rays import origin_direction_of_ray
HOST='127.0.0.1'
PORT=12348
device = "cuda"
n_samples = 150 #change to 1?


s = socket.socket()
print('Socket Created')
s.bind((HOST, PORT))
print('Socket Bound')
s.listen(5)
print('LISTENING')


with open(directory+"MLP_density.json") as d:
	    config_density = json.load(d)

with open(directory+"MLP_color.json") as c:
	    config_color = json.load(c)

config_yml = OmegaConf.load(directory+"default_config.yml")

nerf = NeRF_pipeline(config_yml, config_density, config_color)
sampling  = NeRF_sampling(1, 10,n_samples)
model = NeRF_Network(config_density, config_color)
model.load_state_dict(torch.load(directory+'weights_3.pth'))
model.eval()
fov = 1.7
focal_length = 0.5*config_yml.image_w/np.tan(0.5*fov)

#train_data = scene_dataset(config_yml.image_w, config_yml.image_h, config_yml.json_path, config_yml.image_path)
#train_data_loader = DataLoader(train_data, batch_size=19200, shuffle=False)
i = 0
#for data in train_data_loader:

c, addr = s.accept()
print('Got connection from', addr)
while True:
	data = b""
	mat = loads(c.recv(4096))
	print(mat)
	with(torch.no_grad()):
	        c2w = torch.Tensor(mat)
	        ray_D, ray_O = origin_direction_of_ray(c2w, config_yml.image_w//4, config_yml.image_h//4, focal_length/4.)
	        ray_ds = ray_D.reshape((19200,3)).to(device)
	        ray_os = ray_O.reshape((19200,3)).to(device)
	#	        imgs = data["rgbs"]
	        a = 0
	        rgb_image = []
	        for i in range(0,19200,100):
	                
	        #plt.imshow(img.reshape(120,160,3).detach().cpu().numpy())
	        #plt.show()
	#	                print(i)
	                ray_o = ray_os[i:i+100]
	                ray_d = ray_ds[i:i+100]
	#	                img = imgs[i:i+100]
	        # get coarse samples (n_rays=2048,n_samples=64) (h,w,n_samples)
	                n_coarse = sampling.sample_N_c(ray_o)
	                n_coarse=n_coarse.to(device)
	#	                print(n_coarse.shape)
	                # inputs to the MLP (xyz, view_d)
	                # shape = (n_rays, n_samples, 3)
	                xyz_coarse = (ray_o[..., None, :] + (ray_d[..., None, :] * n_coarse[..., None]))
	#	                print(xyz_coarse.shape)

	                view_dir_coarse =  torch.Tensor.size(xyz_coarse[..., :3])
	                view_dir_coarse = torch.broadcast_to(ray_d[..., None, :], view_dir_coarse)

	                # reshaping the inputs (n_ray*n_sample, 3)
	                xyz_coarse_r = xyz_coarse.reshape((-1,3))
	                view_dir_coarse_r = view_dir_coarse.reshape((-1,view_dir_coarse.shape[-1]))

	                # forward pass to the Neural Network
	                C = xyz_coarse_r.shape[0]

	                #for i in range(0,C,12288):
	                        #MLP network
	                pred_sigma, pred_color = model(xyz_coarse_r, view_dir_coarse_r, device=device)

	                rgbs = pred_color.reshape(xyz_coarse.shape)
	#	                print(rgbs.shape)
	                sigmas = pred_sigma.reshape(xyz_coarse.shape[:-1])

	                # render image (image = n_rays,3 weights = n_rays,n_samples)
	#	                print(rgbs.shape,'rgbs')
	                image_map, weights = render_image(rgbs, sigmas, n_coarse, ray_d, device="cuda")
	                #weights = weights.reshape(weights.shape[0]*weights.shape[1],150)

	                n_mid_points = 0.5 * (n_coarse[..., 1:] + n_coarse[..., :-1])
	                #n_mid_points = n_mid_points.reshape(120*160,150-1)

	                n_fine = sampling.sample_N_f(n_mid_points, weights[...,1:-1],n_samples,device)
	                #n_fine = n_fine.reshape(120,160,n_samples)
	                n_fine = n_fine.detach()

	                n_total, _ = torch.sort(torch.cat([n_coarse, n_fine], -1), -1)

	                xyz_fine = (ray_o[..., None, :] + (ray_d[..., None, :] * n_total[..., None]))
	                #n_rays = xyz_fine.shape[0]
	                view_dir_fine =  torch.Tensor.size(xyz_fine[..., :3])
	                view_dir_fine = torch.broadcast_to(ray_d[..., None, :], view_dir_fine)

	                # reshaping the inputs (n_ray*n_sample, 3)
	                xyz_fine_r = xyz_fine.reshape((-1,3))
	                view_dir_fine_r = view_dir_fine.reshape((-1,view_dir_fine.shape[-1]))

	                pred_sigma_f, pred_color_f = model(xyz_fine_r, view_dir_fine_r, device="cuda")

	                rgbs_f = pred_color_f.reshape(xyz_fine.shape) #(100,100,64,3)
	                sigmas_f = pred_sigma_f.reshape(xyz_fine.shape[:-1])
	                #render image
	                image_map_f, weights = render_image(rgbs_f, sigmas_f, n_total, ray_d, device)
	                rgb_image.append(image_map)
	        
	        final_img = torch.cat(rgb_image,0).view(120,160,3)
	        #plt.imshow(final_img.reshape(120,160,3).detach().cpu().numpy())
	        #plt.show()
	        print('completed')

	        c.send(dumps(final_img.reshape(120,160,3).detach().cpu().numpy()))
	        c.send(b'done')
#ray_d = train_data.scene.ray_dir[38400:57600]

#ray_o = train_data.scene.ray_origin[38400:57600]
#img = train_data.scene.train_imgs[38400:57600]
#plt.imshow(img.reshape(120,160,3).detach().cpu().numpy())
#plt.show()

# data = { "ray_D": ray_d,
#         "ray_O": ray_o,
#         "rgbs": img 
#                       }
      
#image = nerf.coarse_MLP(data)


#ray_d = data["ray_D"].to(device)
#n_rays = self.ray_d.shape[0]

# shape = (batch*3)
#ray_o = data["ray_O"].to(device)

#ray_d = train_data.scene.ray_dir[38400:57600]

#ray_o = train_data.scene.ray_origin[38400:57600]
#img = train_data.scene.train_imgs[38400:57600]
#plt.imshow(img.reshape(120,160,3).detach().cpu().numpy())
#plt.show()

# data = { "ray_D": ray_d,
#         "ray_O": ray_o,
#         "rgbs": img 
#                       }
    
#image = nerf.coarse_MLP(data)


#ray_d = data["ray_D"].to(device)
#n_rays = self.ray_d.shape[0]

# shape = (batch*3)
#ray_o = data["ray_O"].to(device)
