# -*- coding: utf-8 -*-
from data import prepare_data, prepare_model

def tran_data_generator():
    for act_user in prepare_data.get_activity_users():
        yield prepare_data.user_train_data(act_user).next()

recomm_model = prepare_model.generate_recommend_model()
recomm_model.fit_generator(tran_data_generator(), samples_per_epoch=2000, nb_epoch=50)
recomm_model.save_weights('recomm_weight.h5')
