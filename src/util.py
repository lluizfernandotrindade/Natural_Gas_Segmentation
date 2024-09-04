import numpy as np
import tensorflow as tf

def to_patches_2D(patches, patch_size, channel):
        
    patch = patches[0]
    patch = tf.reshape(patch, (patch.shape[0], patch_size, patch_size, channel))
    
    for i in range(1, patches.shape[0]):
        _patch = patches[i]
        _patch = tf.reshape(_patch, (_patch.shape[0], patch_size, patch_size, channel))
        patch = tf.concat((patch, _patch), axis=0)

    return patch


def tf_patches_2D(patches, patch_size, channel):
    patch_list = tf.numpy_function(to_patches_2D, [patches, patch_size, channel], tf.float32)
    return patch_list

def filters_patches_2D(real_patches, generated_patches, mask_patches):
    """
    real_patches: (Batch, h, w, channel)
    generated_patches: (Batch, h, w, channel)
    mask_patches: (Batch, h, w, channel)
    """
    filter_real_patches = []
    filter_generated_patches = []
    
    for i in range(real_patches.shape[0]):
        
        if(mask_patches[i].all()):
            filter_real_patches.append(real_patches[i])
            filter_generated_patches.append(generated_patches[i])
            
    
    return np.ndarray.astype(np.array(filter_real_patches), np.float32), np.ndarray.astype(np.array(filter_generated_patches), np.float32)  
    
    
def tf_filters_patches_2D(real_patches, generated_patches, mask_patches):
    filter_real_patches, filter_generated_patches = tf.numpy_function(filters_patches_2D, [real_patches, 
                                                                                           generated_patches, 
                                                                                           mask_patches], [tf.float32 , tf.float32])
    return filter_real_patches, filter_generated_patches


def torch_gather(x, indices, gather_axis):

    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(tf.cast(gather_locations, dtype=tf.int64))
        else:
            gather_indices.append(tf.cast(all_indices[:, axis], dtype=tf.int64))

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped

def my_numpy_func(x, len_keep):
    x[:, :len_keep] = 0
    return np.ndarray.astype(np.array(x), np.float32)


def tf_function(mask, len_keep): 
    y = tf.numpy_function(my_numpy_func, [mask, len_keep], tf.float32)
    return y

def patchify(patch_embed_size, imgs, channel):
    p = patch_embed_size
    assert imgs.shape[1] == imgs.shape[2] and imgs.shape[2] % p == 0
    
    h = w = imgs.shape[2] // p
    imgs =  tf.transpose(imgs, [0,3,1,2])
    x = tf.reshape(imgs,shape=(imgs.shape[0], channel, h, p, w, p))
    x = tf.cast(tf.einsum("nchpwq->nhwpqc", x),dtype=x.dtype)
    x = tf.reshape(x, shape=(imgs.shape[0], h * w, p**2 * channel))
    
    return x

def tf_patchify(patch_embed_size, imgs, channel):
    imgs = tf.numpy_function(patchify, [patch_embed_size, imgs, channel], tf.float32)
    return imgs

def unpatchify(patch_embed_size, x, channel):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_embed_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = tf.reshape(x, shape=(x.shape[0], h, w, p, p, channel))
    x = tf.einsum("nhwpqc->nchpwq", x)
    imgs = tf.reshape(x, shape=(x.shape[0], channel, h * p, h * p))
    imgs = tf.transpose(imgs, [0,2,3,1])
    
    return imgs

def tf_unpatchify(patch_embed_size, x, channel):
    
    imgs = tf.numpy_function(unpatchify, [patch_embed_size, x, channel], tf.float32)
    
    return imgs