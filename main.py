import pickle

import torch

from networks import nn_registry
from src.metric import Metrics
from src.dataloader import fetch_trainloader,fetch_trainloader2,fetch_trainloader3,fetch_trainloader4
from src import fedlearning_registry
from src.attack import Attacker, grad_inv
from src.compress import compress_registry
from utils import *
from scipy.fft import dct, idct
from quantization import LatticeQuantization, ScalarQuantization
from privacy import Privacy
import torch.nn as nn

class JoPEQ:  # Privacy Quantization class
    def __init__(self, config,gamma,vec_normalization):
        self.vec_normalization = vec_normalization
        dither_var = None
        privacy = False
        quantization = True
        if quantization:
            if config.lattice_dim > 1:
                self.quantizer = LatticeQuantization(config,gamma)
                dither_var = self.quantizer.P0_cov
            else:
                self.quantizer = ScalarQuantization(config,gamma)
                dither_var = (self.quantizer.delta ** 2) / 12
        else:
            self.quantizer = None
        if privacy:
            self.privacy = Privacy(config, dither_var)
        else:
            self.privacy = None

    def divide_into_blocks(self, input, dim=2):
        # Zero pad if needed
        modulo = len(input) % dim
        if modulo:
            pad_with = dim - modulo
            input_vec = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
        input_vec = input.view(dim, -1)  # divide input into blocks
        return input_vec, pad_with,

    def __call__(self, input):
        original_shape = input.shape
        if input.numel() != 1 and input.numel() != 0:
            input = input.view(-1)
            if self.vec_normalization:  # normalize
                input, pad_with = self.divide_into_blocks(input)

            mean = torch.mean(input, dim=-1, keepdim=True)
            std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)

            std = 3 * std
            input = (input - mean) / std

            if self.privacy:
                input = self.privacy(input)

            if self.quantizer is not None:
                input = self.quantizer(input)

            # denormalize
            input = (input * std) + mean

            if self.vec_normalization:
                input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

            input = input.reshape(original_shape)

        return input
def set_gamma(config):
    eta = 1.5
    gamma = 1
    privacy = True
    if privacy:
        if config.lattice_dim == 1:
            gamma += 2 * ((2 / config.epsilon) ** 2)
        else:
            gamma += config.sigma_squared * (config.nu / (config.nu - 2))
    return eta * np.sqrt(gamma)


def set_vec_normalization(config):
    vec_normalization = True
    privacy = False
    quantization = False
    if quantization:
        if config.lattice_dim == 2:
            vec_normalization = True
    if privacy:
        if config.privacy_noise == 't' or config.privacy_noise == 'jopeq_vector':
            vec_normalization = True
    return vec_normalization



def main(config_file):
    
    config = load_config(config_file)
    gamma = set_gamma(config)
    vec_normalization = set_vec_normalization(config)
    output_dir = init_outputfolder(config)
    logger = init_logger(config, output_dir)

    # Load dataset and fetch the data
    train_loader = fetch_trainloader(config, shuffle=True)
    

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx == 0:
            break

    criterion = cross_entropy_for_onehot
    model = nn_registry[config.model](config)
    # model = nn_registry[config.model](config)

    onehot = label_to_onehot(y, num_classes=config.num_classes)
    x, y, onehot, model = preprocess(config, x, y, onehot, model)

    # federated learning algorithm on a single device
    fedalg = fedlearning_registry[config.fedalg](criterion, model, config) 
    grad = fedalg.client_grad(x, onehot)

# CAFL 
    if config.strategy == 'CAFL':
            j = 0
            weights_numbers = {}
            for i in range(len(grad)):
                grad_tensor = grad[i]
                j += 1
                N = grad_tensor.numel()
                weights_numbers[i] = torch.tensor(N)
                M = max(int(config.compression_ratio * N), 1)
                w_dct = dct(grad_tensor.cpu().numpy().reshape((-1, 1)))
                e = j
                y1 = np.zeros_like(w_dct)

                e = j
                if e >= int(N / M) and M != 0:
                    e = e - int(N / M) * int(j / int(N / M))
    
                start = e * M
                end = min((e + 1) * M, len(w_dct))


                y1[start:end, :] = w_dct[start:end, :]
                epsilon_user = config.epsilon + np.zeros_like(y1)
                min_weight = min(y1)
                max_weight = max(y1)
                center = (max_weight + min_weight) / 2
                radius = (max_weight - center) if (max_weight - center) != 0. else 1
                miu = y1 - center
                Pr = (np.exp(config.epsilon) - 1) / (2 * np.exp(config.epsilon))+ np.zeros_like(y1)
                u = np.zeros_like(y1)
                for idx in range(len(y1)):
                    u[idx] = np.random.binomial(1, Pr[idx])

                for idx in range(len(y1)):
                    if u[idx] > 0:
                        y1[idx] = center + miu[idx] * ((np.exp(epsilon_user[idx]) + 1) / (np.exp(epsilon_user[idx]) - 1))
                    else:
                        y1[idx] = center + miu[idx] * ((np.exp(epsilon_user[idx]) - 1) / (np.exp(epsilon_user[idx]) + 1))
                # y1 = [np.array([elem], dtype=np.float32) if not isinstance(elem, np.ndarray) else elem for elem in y1]
                y2 = torch.tensor(y1).to('cuda:0')
                grad[i] = torch.reshape(y2,grad_tensor.shape)
    
    if config.strategy == 'JoPEQ':
        mechanism = JoPEQ(config,gamma,vec_normalization)
        for i in range(len(grad)):
            g0 = grad[i]
            grad[i] = mechanism(g0)

    # gradient postprocessing
    if config.compress != "none":
        compressor = compress_registry[config.compress](config)
        for i, g in enumerate(grad): 
        # this only for resnet18, if you want to apply FLC , please comment if statement
            if i == 20:
                compressed_res = compressor.compress(g)
                grad[i] = compressor.decompress(compressed_res)

    # initialize an attacker and perform the attack 
    attacker = Attacker(config, criterion)
    attacker.init_attacker_models(config)
    recon_data = grad_inv(attacker, grad, x, onehot, model, config, logger)

    synth_data, recon_data = attacker.joint_postprocess(recon_data, y)  
    recon_data = recon_data.detach()      
    # recon_data = synth_data

    # Report the result first 
    logger.info("=== Evaluate the performance ====")
    metrics = Metrics(config)
    snr, ssim, jaccard, lpips = metrics.evaluate(x, recon_data, y ,logger)
    
    logger.info("PSNR: {:.3f} SSIM: {:.3f} Jaccard {:.3f} Lpips {:.3f}".format(snr, ssim, jaccard, lpips))

    save_batch(output_dir, x, recon_data)

    record = {"snr":snr, "ssim":ssim, "jaccard":jaccard, "lpips":lpips}
    with open(os.path.join(output_dir, config.fedalg+".dat"), "wb") as fp:
        pickle.dump(record, fp)

if __name__ == '__main__':
    torch.manual_seed(0)
    main("config.yaml")

