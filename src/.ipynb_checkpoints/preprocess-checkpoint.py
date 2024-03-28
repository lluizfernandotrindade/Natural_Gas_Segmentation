import skimage
import numpy as np


def standarlization_patches(patches_array):
    
    for i in range(patches_array.shape[0]):
        patches_array[i,:,:] = (patches_array[i,:,:] - patches_array[i,:,:].mean()) / patches_array[i,:,:].std()

    return patches_array

def normalize_between_zero_one(
    data
):    
    """Normalizes an nparray to the range [0,+1].
    Args:
        data (nparray): nparray containing the information to be normalized.

    Returns:
        data (nparray): nparray of the same dimension as the input, normalized between [0,+1].
    """
    maxAbsValue = np.abs(np.max(data)) if np.abs(np.max(data)) >= np.abs(np.min(data)) else np.abs(np.min(data))
    data = data / maxAbsValue 
    
    maxValue = 1
    minValue = -1

    return ((data - minValue) / (maxValue - minValue))


def normalize_data(
    data
):    
    """Normalizes an nparray to the range [0,+1].
    Args:
        data (nparray): nparray containing the information to be normalized.

    Returns:
        data (nparray): nparray of the same dimension as the input, normalized between [0,+1].
    """
    maxAbsValue = np.abs(np.max(data)) if np.abs(np.max(data)) > np.abs(np.min(data)) else np.abs(np.min(data))
    return data / maxAbsValue 


def create_patch_2D(data, patch_size=224, step=int(224/2)):
    patch_list = []
    
    for i in range(data.shape[0]):
        inline = data[i].T
        patch_list.append(skimage.util.view_as_windows(inline, (patch_size, patch_size), 
                                                       step=(step, step)))
    
    return np.array(patch_list)


def reshape_patches_list(data):
    
    if data.shape[1] == 1:
        data = data.reshape((data.shape[0],
                             data.shape[2],
                             data.shape[3], 
                             data.shape[4]))
        
        x = []
    
        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                x.append(data[i][j])
            
        data = np.expand_dims(x, axis=-1)
        
    
    elif data.shape[2] == 1:
        data = data.reshape((data.shape[0],
                             data.shape[1],
                             data.shape[3], 
                             data.shape[4]))
        x = []
    
        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                x.append(data[i][j])
            
        data = np.expand_dims(x, axis=-1)
        
        
    else:
        
        x = []
        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                for k in range(data[i][j].shape[0]):
                    x.append(data[i][j][k])
                    
        data = np.expand_dims(x, axis=-1)
    
    return data    

def to_traces_list(data):
         
    x = []
    for i in range(data.shape[0]):
        for j in range(data[i].shape[1]):
                x.append(data[i][:][j])
                    
    data = np.array(x)
    
    return data


def filter_data(true_data, reconstruct, mask):
  
    filter_real_data = []
    filter_reconstruct_data = []
    
    for i in range(mask.shape[0]):
        
        if(mask[i,:].all() == 0):
            filter_real_data.append(true_data[i,:])
            filter_reconstruct_data.append(reconstruct[i,:]) 
            
    return np.array(filter_real_data), np.array(filter_reconstruct_data)