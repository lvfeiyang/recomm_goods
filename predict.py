# -*- coding: utf-8 -*-
from data import prepare_data, prepare_model

recomm_model = prepare_model.generate_recommend_model()
recomm_model.load_weights('recomm_weight.h5')

for predict_user in prepare_data.get_activity_users(True):
    for predict_data, goods_id in prepare_data.user_predict_data(predict_user):
        result = recomm_model.predict(predict_data)#predict_proba
        if 2 == result:
            print goods_id
            break
    break
