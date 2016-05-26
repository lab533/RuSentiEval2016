from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Model
import numpy as  np


def build_model(max_features, weights=None, embedding_dim=300):    
    '''
    Return model of CNN for work with ancestors and siblings
    max_features
    '''
    
    # linear order and ancestors input    
    ancestors = Input(shape=(250, ), dtype='int32', name="ancestors")
    siblings2 = Input(shape=(100, ), dtype='int32', name="siblings2")
    siblings3 = Input(shape=(150, ), dtype='int32', name="siblings3")
    siblings3_ = Input(shape=(150, ), dtype='int32', name="siblings3_")
    siblings4 = Input(shape=(200, ), dtype='int32', name="siblings4")
    siblings4_ = Input(shape=(200, ), dtype='int32', name="siblings4_")
    sib_f_lens = [2, 3, 3, 4, 4]
    siblings = [siblings2, siblings3, siblings3_, siblings4, siblings4_]

    
    w2v_layer = Embedding(output_dim=embedding_dim, 
                          input_dim=max_features,
                          weights=weights,
                          name='embeddings')

    ancestors_vecs = w2v_layer(ancestors)
    siblings_vecs = [w2v_layer(sibling) for sibling in siblings]
    
    anc_f_lens = [3, 4, 5]
    ancestors_conv = [Convolution1D(nb_filter=100, 
                                    filter_length=f_len, 
                                    subsample_length=5,
                                    activation='relu', name='ancestors_conv_' + str(f_len))(ancestors_vecs) 
                  for f_len in anc_f_lens]   
    
    siblings_conv = [Convolution1D(nb_filter=100,
                                   filter_length=f_len,
                                   subsample_length=f_len,
                                   activation='relu',
                                   name='siblings_conv_' + str(f_len)+ '_' + str(idx))(siblings_vecs[idx])
                 for idx, f_len in enumerate(sib_f_lens)]


    anc_sib_conv = ancestors_conv + siblings_conv

    # Pooling
    max_pool_layer = MaxPooling1D(pool_length=50, name='maxpool')
    anc_sib_pool = [max_pool_layer(conv) for conv in anc_sib_conv]

    # Flatten
    flatten_layer = Flatten(name='flatten')
    anc_sib_flat = [flatten_layer(pool) for pool in anc_sib_pool]

    # Merge
    convolve_merged = merge(anc_sib_flat, mode='concat')

    # MLP
    dropped = Dropout(0.25)(convolve_merged)
    
    mlp = Dense(200, activation='relu', name='dense_relu')(dropped)
    dropped = Dropout(0.5)(mlp)
    mlp = Dense(3, activation='softmax', name='dense_softmax')(dropped)
    model = Model(input=[ancestors] + siblings, output=[mlp])
    model.compile(loss='categorical_crossentropy',
              optimizer='adagrad', metrics=['accuracy'])
    
    return model
