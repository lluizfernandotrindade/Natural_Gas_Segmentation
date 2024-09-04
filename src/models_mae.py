import tensorflow as tf
import keras.backend as K
from src.vision_transformer import PatchEmbed, Decoder, Encoder
from src.pos_embed import get_2d_sincos_pos_embed
from src.util import torch_gather, tf_function, tf_patchify, tf_unpatchify, tf_patches_2D, tf_filters_patches_2D
import numpy as np

class MaskedAutoencoder(tf.keras.Model):
    """ Masked Autoencoder with Transformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=384, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., mask_ratio=0.75, batch_size=8):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.channel = in_chans
        self.batch_size = batch_size
        self.patch_size = patch_size
        
        self.patch_embed = PatchEmbed(img_size, 
                                      patch_size, 
                                      in_chans, 
                                      embed_dim
                                     )
        
        self.num_patches = self.patch_embed.num_patches

        
        self.cls_token = tf.Variable(name="cls_token",
                                     initial_value = tf.random.normal(
                                         shape=(1, 1, embed_dim), dtype='float32'), 
                                     trainable=True
                                    )
        self.pos_embed = tf.Variable(name="pos_embed", 
                                     initial_value=tf.zeros((1, self.num_patches + 1, embed_dim)), 
                                     trainable=True)  # fixed sin-cos embedding


        self.encoder_block = Encoder(embed_dim,
                                     num_heads,
                                     mlp_ratio,
                                     depth
                                     )
        
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.mask_token = tf.Variable(name="mask_token",
                                     initial_value = tf.random.normal(
                                         shape=(1, 1, decoder_embed_dim), dtype='float32'), 
                                     trainable=True
                                    )

        self.decoder_pos_embed = tf.Variable(name="decoder_pos_embed",
                                             initial_value=tf.zeros((1, self.num_patches + 1, decoder_embed_dim))
                                            )  # fixed sin-cos embedding

        self.decoder_blocks = Decoder(embed_dim,
                                      patch_size,
                                      decoder_embed_dim,
                                      decoder_num_heads,
                                      mlp_ratio,
                                      decoder_depth,
                                      self.channel
                                     )

        self.decoder_embed = tf.keras.layers.Dense(decoder_embed_dim, activation='linear')
           
        self.mask_ratio = mask_ratio        

        # --------------------------------------------------------------------------

        self.initialize_weights()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = tf.expand_dims(get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                               int(self.patch_embed.num_patches**.5), cls_token=True
                              ), axis=0)
                
        
        self.decoder_pos_embed = tf.expand_dims(get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                       int(self.patch_embed.num_patches**.5), cls_token=True
                                      ), axis=0)
        
    

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        imgs = tf_patchify(self.patch_embed.patch_size[0], imgs, self.channel)
        
        return imgs

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """      
        imgs = tf_unpatchify(self.patch_embed.patch_size[0], x, self.channel)
        
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        if N != None:
            self.batch_size = N 
    
        noise = tf.random.normal([self.batch_size, L])
    
        # sort noise for each sample
        ids_shuffle = tf.argsort(noise, axis=1)  # ascend: small is keep, large is remove
        ids_restore = tf.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_keep = tf.expand_dims(ids_keep, axis=-1)
        ids_keep = tf.repeat(ids_keep, repeats=D, axis=-1)

        x_masked = torch_gather(x, ids_keep, 1)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = tf.ones([self.batch_size, L])
        #mask[:, :len_keep] = 0

        mask = tf_function(mask, len_keep)

        # unshuffle to get the binary mask
        mask = torch_gather(mask, ids_restore, 1)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        
        cls_tokens = tf.broadcast_to(cls_token, [x.shape[0], cls_token.shape[1], cls_token.shape[2]])
        
        x = tf.concat((cls_tokens, x), axis=1)

        # apply Transformer blocks
        x = self.encoder_block(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = tf.repeat(self.mask_token, ids_restore.shape[1] + 1 - x.shape[1], axis=1)
        mask_tokens = tf.broadcast_to(mask_tokens, [x.shape[0], mask_tokens.shape[1], mask_tokens.shape[2]])

        
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)  # no cls token
        
        ids_restore = tf.expand_dims(ids_restore, axis=-1)
        ids_restore = tf.repeat(ids_restore, repeats=x.shape[2], axis=-1)
        
        x_ = torch_gather(x_, ids_restore, 1)  # unshuffle
        x = tf.concat([x[:, :1, :], x_], axis=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def compile(self, optimizer, loss_fn):
            super().compile()
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            
    def train_step(self, imgs):
                 
        with tf.GradientTape() as tape:
            latent, mask, ids_restore = self.forward_encoder(imgs, 
                                                             self.mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            
            target = self.patchify(imgs)
            loss = self.loss_fn(target,
                                pred, 
                                mask)
       
        # Apply gradients.
        train_vars = [
            self.encoder_block.trainable_variables,
            self.decoder_blocks.trainable_variables,
            self.decoder_embed.trainable_variables,
            self.patch_embed.trainable_variables,
        ]
        
        grads = tape.gradient(loss, train_vars)
        #del tape

        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, imgs):
        
        with tf.GradientTape() as tape:
            latent, mask, ids_restore = self.forward_encoder(imgs, 
                                                             self.mask_ratio
                                                            )
            pred = self.forward_decoder(latent, 
                                        ids_restore
                                       )  # [N, L, p*p*3]
            target = self.patchify(imgs)

            loss = self.loss_fn(target, 
                                pred, 
                                mask)
              
        
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": self.loss_tracker.result()}
