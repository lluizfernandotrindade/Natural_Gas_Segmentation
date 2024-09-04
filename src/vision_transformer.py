import tensorflow as tf

   
class PatchEmbed(tf.keras.layers.Layer):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=384,
            norm_layer=False,
            flatten=True,
    ):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]        
        self.flatten = tf.keras.layers.Reshape(target_shape=(-1, embed_dim), name="flatten") if flatten else flatten

        self.proj = tf.keras.layers.Conv2D(embed_dim, 
                                           kernel_size=patch_size, 
                                           strides=patch_size, 
                                           name="projection")
        
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layernorm")

    def call(self, x):
        
            B, H, W, C = x.shape
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
            x = self.proj(x)

            if self.flatten:
                x = self.flatten(x)
                
            x = self.norm(x)
                
            return x



class Blocks(tf.keras.layers.Layer):
    def __init__(
        self, embed_dim, mlp_dim, num_heads, dropout_rate=0.0, attention_dropout_rate=0.0, **kwargs
    ):
        super(Blocks, self).__init__(**kwargs)
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=(embed_dim // num_heads), 
            dropout=attention_dropout_rate
        )
        
        self.dense_0 = tf.keras.layers.Dense(
            units=embed_dim*mlp_dim,
            activation="gelu",
        )
        self.dense_1 = tf.keras.layers.Dense(
            units=embed_dim,
            activation="linear",
        )

        self.dropout_0 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.norm_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, inputs, training):
        # Attention block
        x = self.norm_0(inputs)
        x = self.mha(x,x)
        #x = self.dropout_0(x, training=training)
        x = x + inputs

        # MLP block
        y = self.norm_1(x)
        y = self.dense_0(y)
        y = self.dense_1(y)
        #y = self.dropout_1(y, training=training)

        return x + y



def Encoder2(embed_dim,
            num_heads,
            mlp_ratio,
            depth
           ):
    
    inputs = tf.keras.layers.Input((None, embed_dim))
    
    for _ in range(depth):

        # Create a multi-head attention layer.        
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=(embed_dim // num_heads), dropout=0.1
        )(inputs, inputs)
        
        

        # Skip connection 1.
        skip = x + inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(skip)
        
        # Feed Forward Part
        x = tf.keras.layers.Conv1D(int(embed_dim*mlp_ratio), kernel_size=1, activation='gelu')(x)
        x = tf.keras.layers.Conv1D(embed_dim, kernel_size=1)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
       # Skip connection 2.

        x = x + skip    
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)  

        
    
    model = tf.keras.Model(inputs, x, name="encoder_block")
    
    return model

def Encoder(embed_dim,
            num_heads,
            mlp_ratio,
            depth
           ):
    
    inputs = tf.keras.layers.Input((None, embed_dim))
    x = inputs
    for _ in range(depth):

        # Create a multi-head attention layer.  
        
        
        att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=(embed_dim // num_heads), dropout=0.1
        )(x, x)
        
        

        # Skip connection 1.
        skip = x + att
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(skip)
        
        # Feed Forward Part
        x = tf.keras.layers.Dense(int(embed_dim*mlp_ratio), activation='gelu')(x)
        x = tf.keras.layers.Dense(embed_dim)(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # Skip connection 2.
        x = skip + x     
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        
        
    
    model = tf.keras.Model(inputs, x, name="encoder_block")
    
    return model
    
def Decoder(embed_dim,
            patch_size,
            decoder_embed_dim,
            num_heads,
            mlp_ratio,
            depth,
            in_channel
           ):
    
    inputs = tf.keras.layers.Input((None, decoder_embed_dim))
    x = inputs
    
    for _ in range(depth):
        #x = TransformerBlock(num_heads=num_heads, mlp_dim=int(embed_dim * mlp_ratio), dropout=0.1) (x) 
        # Layer normalization 1.
                # Create a multi-head attention layer.
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=(inputs.shape[-1] // num_heads), dropout=0.0
        )(x, x)

        # Skip connection 1.
        x = x + inputs

        # Layer normalization 2.
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        y = tf.keras.layers.Dense(int(decoder_embed_dim * mlp_ratio), activation='gelu')(y)
        #y = tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x, approximate=False))(y)
        y = tf.keras.layers.Dense(inputs.shape[-1], activation='linear')(y)
        y = tf.keras.layers.Dropout(0.0)(y)

        # Skip connection 2.
        #x = tf.keras.layers.Add()([x3, x2])
        x = x + y
        #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(units=patch_size**2*in_channel, activation="linear", name="output_layer")(x)
    model = tf.keras.Model(inputs, x, name="mae_decoder")
    model.summary()
    
    return model
