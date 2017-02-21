# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, SpatialDropout2D, Merge
from keras.layers import Convolution2D, MaxPooling2D

def generate_recommend_model():
    model_goods_img = Sequential()

    model_goods_img.add(Convolution2D(300, 15, 15, input_shape=(200,200,3)))
    model_goods_img.add(Activation('relu'))
    model_goods_img.add(MaxPooling2D(pool_size=(2, 2)))
    model_goods_img.add(SpatialDropout2D(0.25))

    model_goods_img.add(Convolution2D(500, 10, 10))
    model_goods_img.add(Activation('relu'))
    model_goods_img.add(MaxPooling2D(pool_size=(2, 2)))
    model_goods_img.add(SpatialDropout2D(0.25))

    model_goods_img.add(Convolution2D(1000, 3, 3))
    model_goods_img.add(Activation('relu'))
    model_goods_img.add(MaxPooling2D(pool_size=(2, 2)))
    model_goods_img.add(SpatialDropout2D(0.25))

    model_goods_img.add(Flatten())
    model_goods_img.add(Dense(1024))
    model_goods_img.add(Activation('relu'))
    model_goods_img.add(Dropout(0.2))
    model_goods_img.add(Dense(100))
    model_goods_img.add(Activation('relu'))

    #goods:list_price currency current_price title(20) desc(100) discount gender
    #cny_price original_site_id product_type_id category_id brand_id

    model_goods_desc = Sequential()
    model_goods_desc.add(Dense(40, input_dim=120))
    model_goods_desc.add(Activation('relu'))
    model_goods_desc.add(Dropout(0.2))
    model_goods_desc.add(Dense(10))
    model_goods_desc.add(Activation('relu'))

    model_goods_info = Sequential()
    model_goods_info.add(Dense(10, input_dim=10)) #50
    model_goods_info.add(Activation('relu'))

    #user: head name create_time(sub now) channels
    #viewed_items: brand site cny_price gender discount category_id product_type_id sourceCode

    model_user_view = Sequential()
    model_user_view.add(Dense(1000, input_dim=100*8))
    model_user_view.add(Activation('relu'))
    model_user_view.add(Dropout(0.2))
    model_user_view.add(Dense(500))
    model_user_view.add(Activation('relu'))
    model_user_view.add(Dropout(0.2))
    model_user_view.add(Dense(50))
    model_user_view.add(Activation('relu'))

    model_user_info = Sequential()
    model_user_info.add(Dense(5, input_dim=3))
    model_user_info.add(Activation('relu'))

    user_goods_merge = Merge([model_user_info, model_user_view, model_goods_info, model_goods_desc, model_goods_img], mode='concat')

    final_model = Sequential()
    final_model.add(user_goods_merge)
    final_model.add(Dense(150))
    final_model.add(Activation('relu'))
    final_model.add(Dropout(0.1))
    final_model.add(Dense(100))
    final_model.add(Activation('relu'))
    final_model.add(Dropout(0.1))
    final_model.add(Dense(10))
    final_model.add(Activation('relu'))
    final_model.add(Dropout(0.1))
    final_model.add(Dense(3, activation='softmax'))

    # final_model.summary()

    final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return final_model

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
