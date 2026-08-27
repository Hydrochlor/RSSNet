import numpy as np
import torch
import os
import random
import argparse
import glob

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)    
    random.seed(seed)    
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True

def generate_red_noise(length):
    white_noise = torch.randn(length)
    cumulative_sum = torch.cumsum(white_noise, dim=0)
    red_noise = cumulative_sum / torch.max(torch.abs(cumulative_sum))
    return red_noise

def add_noise_with_red_and_white(waveform, white_noise_waveform, snr_dB):
    red_noise_waveform = generate_red_noise(len(waveform))
    combined_noise = red_noise_waveform + white_noise_waveform
    signal_power = torch.mean(waveform**2)
    noise_power = torch.mean(combined_noise**2)
    snr_linear = 10 ** (snr_dB / 10)
    scale_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noisy_waveform = waveform + scale_factor * combined_noise
    return noisy_waveform

def addnoise(name, ref_path, list_path, save_path):
    origin_spec = np.loadtxt(ref_path)
    spec_list = np.loadtxt(list_path)
    mix_spec = []
    for i in range(len(spec_list)):
        label1 = int(spec_list[i][0])
        label2 = int(spec_list[i][1])
        factor = spec_list[i][2]
        mix = torch.Tensor(origin_spec[label1] * factor + origin_spec[label2] * (1 - factor))
        snr_dB = np.random.uniform(10, 20)
        noise = torch.randn(1024)
        mix_noise = add_noise_with_red_and_white(mix, noise, snr_dB)
        mix_spec.append(mix_noise.numpy())
    mix_spec = np.array(mix_spec)
    np.savetxt(save_path, mix_spec)
    print(f"{name}: mix_spec successfully saved.")
    print("shape of mix_spec:", mix_spec.shape)

def getlistpath(path, mode):
    pattern = os.path.join(path, os.path.join(mode, 'mix_2_spec*'))
    matched_files = glob.glob(pattern)
    if matched_files:
        match = matched_files[0]
        print(f"Find {match}")
        return match
    else:
        raise RuntimeError("No such list file.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for generating dataset')
    parser.add_argument('--name', type=str, help='Name of dataset.')
    parser.add_argument('--path', type=str, help='Path to dataset.')
    parser.add_argument('--seed', type=int, help='Random seed.')
    args = parser.parse_args()
    setup_seed(args.seed)
    ref_path = os.path.join(args.path, "prep_spec.txt")
    train_path = getlistpath(args.path, "train")
    train_save_path = os.path.join(args.path, os.path.join("train", args.name + '_rednoise_10-20_mix_train.txt'))
    addnoise(args.name + ' Train', ref_path, train_path, train_save_path)

    val_path = getlistpath(args.path, "val")
    val_save_path = os.path.join(args.path, os.path.join("val", args.name + '_rednoise_10-20_mix_val.txt'))
    addnoise(args.name + ' Val', ref_path, val_path, val_save_path)

    test_path = getlistpath(args.path, "test")
    test_save_path = os.path.join(args.path, os.path.join("test", args.name + '_rednoise_10-20_mix_test.txt'))
    addnoise(args.name + ' Test', ref_path, test_path, test_save_path)

    print(f"Finished generating {args.name}-2Mix dataset.")
    

