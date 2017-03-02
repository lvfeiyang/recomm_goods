# -*- coding: utf-8 -*-
from data import prepare_data, prepare_model

def tran_data_generator():
    for act_user in prepare_data.get_activity_users():
        try:
            yield prepare_data.user_train_data(act_user).next()
        except Exception as e:
            logging.error("%s user have bad train data: %s" % (act_user, e))
            continue

recomm_model = prepare_model.generate_recommend_model()
recomm_model.fit_generator(tran_data_generator(), samples_per_epoch=200, nb_epoch=50)
recomm_model.save_weights('recomm_weight.h5')
