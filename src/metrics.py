import cv2
import math
import numpy as np
from scipy import signal

import src.preprocess

def SSIM(img1,
         img2
        ):
    
    K = [0.01, 0.03]
    L = 1
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T
     
    M,N = np.shape(img1)

    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    img1 = np.float64(img1)
    img2 = np.float64(img2)
 
    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim, ssim_map

def PSNR(original, 
         compressed
        ):
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def calculate_metrics(test_data, 
                      model, 
                      masked_ration,
                      out_dir
                     ):
    ssim_metric = []
    psnr_metric = []
    ssim_metric_filtered = []
    psnr_metric_filtered = []
    
    for i in range(test_data.shape[0]):
        true_data = test_data[i]
  
        
        masked_data, _ = model.apply_mask(np.expand_dims(true_data, axis=0), 
                                            masked_ration)
        
        x_data_ = true_data*masked_data
        reconstruct = model.generator.predict(np.expand_dims(x_data_, axis=0))
        reconstruct = np.squeeze(reconstruct, axis=0) 
        
        ssim_value, _ = SSIM(true_data, reconstruct)
        psnr = PSNR(true_data, reconstruct)
        
        ssim_metric.append(ssim_value)
        psnr_metric.append(psnr)
        
        
        filter_real_data, filter_reconstruct_data = preprocess.filter_data(true_data,
                                                                           reconstruct,
                                                                           masked_data.numpy()
                                                                          )
        ssim_value_, _ = SSIM(filter_real_data, filter_reconstruct_data)
        psnr_ = PSNR(filter_real_data, filter_reconstruct_data)
        
        save_metrics_individual_report(out_dir,
                                       ssim_value,
                                       ssim_value_,
                                       psnr,
                                       psnr_,
                                       dataset_name=str(i) 
                                      )
        
        ssim_metric_filtered.append(ssim_value_)
        psnr_metric_filtered.append(psnr_)
        
        
    
    
    return np.mean(ssim_metric), np.mean(psnr_metric), np.mean(ssim_metric_filtered), np.mean(psnr_metric_filtered)

def save_metrics_individual_report(out_dir, 
                             ssim_metric,
                             ssim_metric_filtered,
                             psnr_metric,
                             psnr_metric_filtered,
                             dataset_name
                            ):
    
    with open(out_dir + "/individual_test_report.txt", "a") as f:
            f.writelines([
                'Test patch ' + str(dataset_name) + '\n',
                '   SSIM: {} \n'.format(ssim_metric),
                '   SSIM Filtered: {} \n'.format(ssim_metric_filtered),
                '   PSNR: {} \n'.format(psnr_metric),
                '   PSNR Filtered: {} \n'.format(psnr_metric_filtered),
                '\n'
            ])
            
            
def save_metrics_report_mean(out_dir, 
                             ssim_metric,
                             ssim_metric_filtered,
                             psnr_metric,
                             psnr_metric_filtered,
                             dataset_name
                            ):
    
    with open(out_dir + "/TestReport.txt", "a") as f:
            f.writelines([
                'Dataset ' + str(dataset_name) + '\n',
                '   SSIM: {} \n'.format(ssim_metric),
                '   SSIM Filtered: {} \n'.format(ssim_metric_filtered),
                '   PSNR: {} \n'.format(psnr_metric),
                '   PSNR Filtered: {} \n'.format(psnr_metric_filtered),
                '\n'
            ])