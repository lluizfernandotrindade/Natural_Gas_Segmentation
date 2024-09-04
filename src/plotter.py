import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.preprocess 
import src.metrics

def plot_result(x_test,
                model,
                masked_ration=0.75
               ):

    x = src.preprocess.standarlization(x_test)
    latent, mask, ids_restore = model.forward_encoder(x, 
                                                      masked_ration
                                                     )
    y = model.forward_decoder(latent, 
                              ids_restore
                             )  # [N, L, p*p*3]

    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.repeat(mask, repeats=model.patch_embed.patch_size[0]**2 *model.channel, axis=-1)

    y = model.unpatchify(y)
    mask = model.unpatchify(mask)

    idx = np.random.choice(x_test.shape[0])
    print(f"\nIdx chosen: {idx}")
    original_image = x[idx]
    reconstructed_image = y[idx].numpy()
    mask_ = mask[idx].numpy()

    # masked image
    im_masked = original_image * (1 - mask_)
    visible = original_image * (1 - mask_) + reconstructed_image * mask_
    view_masked_data = 0.5*(np.ones(mask_.shape) - (1 - mask_))
    
    ssim_index, ssim_map = src.metrics.SSIM(np.squeeze(original_image, axis=-1),
                                        np.squeeze(visible, axis=-1)
                                       )

    fig, axs = plt.subplots(5, 1, figsize=(15, 20))

    im1 = axs[0].imshow(original_image, cmap='gray')
    im2 = axs[1].imshow(im_masked + view_masked_data, cmap='gray')
    im3 = axs[2].imshow(reconstructed_image, cmap='gray')
    im4 = axs[3].imshow(visible, cmap='gray')
    im5 = axs[4].imshow(ssim_map, vmin=0, vmax=1, cmap='jet_r')

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    divider2 = make_axes_locatable(axs[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2, orientation='vertical')

    divider3 = make_axes_locatable(axs[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3, orientation='vertical')
    
    divider4 = make_axes_locatable(axs[3])
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, cax=cax4, orientation='vertical')

    divider5 = make_axes_locatable(axs[4])
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im5, cax=cax5, orientation='vertical')

    axs[0].axis('off')
    axs[0].set_title('Original ID = ' + str(idx))

    axs[1].axis('off')
    axs[1].set_title('Masked Ration = ' + str(masked_ration))

    axs[2].axis('off')
    axs[2].set_title('Reconstructed')
    
    axs[3].axis('off')
    axs[3].set_title('Visible')

    axs[4].axis('off')
    axs[4].set_title('SSIM Map')

    plt.show()