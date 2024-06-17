import torch
import netarch
import tiffile
import json
import argparse
import utils
import numpy as np
import dataset
from tqdm import tqdm
import os


def train(config, noisy_data):

    lr = config["lr"]
    gpu = config["gpu"]
    epoch = config["epoch"]
    batch_size = config["batch_size"]
    patch_size = config["patch_size"]

    s = config["s"]
    M = config["M"]

    DEVICE = torch.device('cuda:{}'.format(str(gpu) if torch.cuda.is_available() else 'cpu'))
    denoised_res = np.zeros_like(noisy_data)

    Subsampler = utils.PD_sampler(s, DEVICE)

    train_data = noisy_data[None]
    train_loader = dataset.loader_train(train_data.repeat(40, axis = 0), patch_size, batch_size)
    val_loader = dataset.loader_val(train_data, batch_size=1)

    model = netarch.Noise2SR(in_c = 1, out_c = 1, feature_dim = 128, scale_factor = s).to(DEVICE)
    loss_fun = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    train_loop = tqdm(range(epoch), colour = 'green', leave=False, ncols = 100)

    # Optimizing Network
    for e in train_loop:
        model.train()
        loss_train = 0
        for i, (x1) in enumerate(train_loader):

                x1 = x1.to(DEVICE).float()
                x1_in, mask = Subsampler.sample_img(x1, 1)
                mask = mask.to(DEVICE)
                img_pred = model(x1_in)
                loss = loss_fun(img_pred[mask == 1], x1[mask == 1])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record and print loss
                loss_train += loss.item()
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        train_loop.set_description(f'Epoch [{e + 1}/{epoch}] loss:{loss_train/len(train_loader):.6f}')
    
    # Inference the denoised result
    denoised_res = np.zeros_like(noisy_data)

    model.eval()
    with torch.no_grad():
        for i, (x1) in enumerate(val_loader):
            x1 = x1.to(DEVICE).float().unsqueeze(1)
            for _ in range(M):
                 x1_in, mask = Subsampler.sample_img(x1, 1)
                 denoised_res += model(x1_in).cpu().numpy().squeeze()

    denoised_res = denoised_res / M
    denoised_res = np.array(denoised_res, dtype = np.float32)
    model_weight = model.state_dict()
    return denoised_res, model_weight
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, default='config/simulated_PG.json', dest='config_path',
                        help='the file path of input train noisy data')
    parser.add_argument('-img_path', type=str, default='demo_data/PtCeO2_simulated/1.tif', dest='img_path')
    parser.add_argument('-a', type=float, default=0.05, dest='a')
    parser.add_argument('-b', type=float, default=0.02, dest='b')

    args = parser.parse_args()
    config_path = args.config_path
    a = args.a
    b = args.b

    

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    img_path = args.img_path
    img_index = img_path.split('/')[-1].split('.')[0]
    save_path = config["file"]["save_path"] + f'PG_{a:.2f}_{b:.2f}/'
    model_save_path = config["file"]["model_save_path"]+ f'PG_{a:.2f}_{b:.2f}/'

    for path in [save_path, model_save_path]:
         if os.path.exists(path) == False:
              os.makedirs(path)

    train_config = config['train']
    print(f'Simulated Poission Gaussian denoising with {a:.2f}/{b:.2f}')

    gt_data = tiffile.imread(img_path)
    noisy_data = utils.add_gaussian_poission_noise(gt_data, a, b)

    psnr_index, ssim_index = utils.cal_psnr(gt_data, noisy_data), utils.cal_ssim(gt_data, noisy_data)
    print(f'Noisy/GT PSNR/SSIM:{psnr_index:.2f}/{ssim_index:.4f}')


    denoised_res, model_weight = train(train_config, noisy_data)

    psnr_index, ssim_index = utils.cal_psnr(gt_data, denoised_res), utils.cal_ssim(gt_data, denoised_res)
    print(f'Denoised/GT PSNR/SSIM:{psnr_index:.2f}/{ssim_index:.4f}')

    tiffile.imwrite(save_path + img_index + '_denoised_res.tif', denoised_res)
    torch.save(model_weight, model_save_path + img_index +'_model_weight.pth')
