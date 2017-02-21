# -*- coding: utf-8 -*-
import time
import logging
import pymysql
import random
# import urllib
import cStringIO
import numpy as np
from bson.objectid import ObjectId
from pymongo import MongoClient
from urllib import quote_plus, urlopen
from elasticsearch import Elasticsearch
from keras.preprocessing import text as kp_text, image as kp_image, sequence as kp_sequence

def _connect_mysql(db):
    return pymysql.connect(host='10.66.131.23', user='root', password='yiya1504!!', db=db)

def _connect_mongodb():
    url = "mongodb://%s:%s@%s" % (quote_plus('yiya'), quote_plus('yiya1504!!'), '10.105.18.199:27017')
    return MongoClient(url)
    # client = MongoClient(url)#'mongodb://localhost:27017/')

mysql_brand_map = {}
def _get_brand_map():
    global mysql_brand_map
    if not mysql_brand_map:
        connection = _connect_mysql('super_mammy_shop')
        try:
            with connection.cursor() as cursor:
                sql = "SELECT `id`,`name` FROM `brand`"
                cursor.execute(sql)
                results = cursor.fetchall()
                for res in results:
                    mysql_brand_map[res[1]] = res[0]
        finally:
            connection.close()
    return mysql_brand_map

mysql_site_map = {}
def _get_site_map():
    global mysql_site_map
    if not mysql_site_map:
        connection = _connect_mysql('super_mammy_shop')
        try:
            with connection.cursor() as cursor:
                sql = "SELECT `id`,`name` FROM `original_site`"
                cursor.execute(sql)
                results = cursor.fetchall()
                for res in results:
                    mysql_site_map[res[1]] = res[0]
        finally:
            connection.close()
    return mysql_site_map

gender_dict = {'women':1, 'men':2, 'boys':3, 'girls':4}
def _map_user_view(hits):
    hits_source = hits['_source']
    brand_id = _get_brand_map().get(hits_source['brand_name'], 0)#[hits_source['brand_name']]
    site_id = _get_site_map().get(hits_source['site'], 0)
    source_code = int(hits_source['sourceCode']) if hits_source['sourceCode'] else 0
    return [brand_id, site_id, hits_source['cny_price'], gender_dict.get(hits_source['gender'], 0), hits_source['discount'], hits_source['category_id'], hits_source['product_type_id'], source_code]

def _no_view_goods(relation_time):
    relation_time = relation_time.strftime("%y%m%d")
    rand = random.randint(100, 999)
    time_rand = relation_time + rand
    client = _connect_mongodb()
    mongo_goods = client.shiji_shop.goods.find_one({'status':3, 'rand_order':{'$gte':time_rand}})
    if not mongo_goods:
        mongo_goods = client.shiji_shop.goods.find_one({'status':3, 'rand_order':{'$lte':time_rand}})
    return _goods_train_data(mongo_goods)

channel_dict = {'appstore':1, 'yingyongbao':2, 'wandoujia':3, 'baidu':4, 'huawei':5, 'official':6, 'meizu':7, 'xiaomi':8, 'sanxing':9}
def _user_inherent_info(user_id):
    # url = "mongodb://%s:%s@%s" % (quote_plus('yiya'), quote_plus('yiya1504!!'), '10.105.18.199:27017')
    client = _connect_mongodb()# MongoClient(url)#'mongodb://localhost:27017/')
    mongo_user = client.super_mammy.users.find_one({'user_id':user_id})
    # mongo_user['name'],
    time_diff = time.time() - time.mktime(time.strptime(mongo_user['create_time'], '%Y-%m-%d %H:%M:%S'))
    # channel_dict = {'appstore':1, 'yingyongbao':2, 'wandoujia':3, 'baidu':4, 'huawei':5, 'official':6, 'meizu':7, 'xiaomi':8, 'sanxing':9}
    user_info = np.array([[time_diff, channel_dict.get(mongo_user.get('channel', 'appstore'), 0), mongo_user.get('like_count', 0)]])
    user_info = user_info.astype('float32')
    logging.info('user_info %s:' % str(user_info.shape))
    logging.info(user_info)
    return user_info, mongo_user.get('goodses', [])

def _user_recent_view(user_id, view_time=None):
    if view_time is None:
        view_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    else:
        view_time = view_time.strftime("%Y-%m-%d")
    es = Elasticsearch(['10.105.78.165:9200','10.105.1.33:9200'])
    body = {"query":{"filtered":{"filter":{
        "bool":{"must":[{"term":{"user_id":user_id}},{"range":{"@timestamp":{"lte":view_time}}}]}
        }}},"sort":{"@timestamp":{"order":"desc"}},"size":100}
    res = es.search(index='logstash-action', doc_type='goods-detail', body=body)
    user_view_detail = np.array(map(_map_user_view, res['hits']['hits']))
    miss_line = 100 - user_view_detail.shape[0]
    if miss_line:
        user_view_detail = np.vstack((user_view_detail, np.zeros((miss_line, 8))))
    user_view_detail = user_view_detail.astype('float32').reshape(1, 100*8)
    logging.info('user_view_detail %s:' % str(user_view_detail.shape))
    logging.info(user_view_detail)
    return user_view_detail

def _url_to_image(url):
    url += '?imageView2/1/w/200/h/200'
    resp = urlopen(url)
    image_buf = cStringIO.StringIO(resp.read())
    image = kp_image.load_img(image_buf)
    return kp_image.img_to_array(image)

def _goods_train_data(goods):
    if isinstance(goods, basestring):
        client = _connect_mongodb()
        mongo_goods = client.shiji_shop.goods.find_one({'_id':ObjectId(goods)})
    else:
        mongo_goods = goods
    if mongo_goods:
        goods_info = np.array([[mongo_goods['list_price'], mongo_goods['currency'], mongo_goods['current_price'],
            mongo_goods['discount'], gender_dict.get(mongo_goods['gender'], 0), mongo_goods['cny_price'],
            mongo_goods['original_site_id'], mongo_goods['product_type_id'], mongo_goods['category_id'], mongo_goods['brand_id']]])
        goods_info = goods_info.astype('float32')
        logging.info('goods_info %s:' % str(goods_info.shape))
        logging.info(goods_info)

        goods_desc_vector_org = kp_text.one_hot(mongo_goods['desc'].encode('utf-8'), 512)
        goods_desc_vector = kp_sequence.pad_sequences([goods_desc_vector_org], maxlen=100, padding='post', truncating='post') #goods_desc_vector_cut
        goods_title_vector_org = kp_text.one_hot(mongo_goods['title'].encode('utf-8'), 512)
        goods_title_vector = kp_sequence.pad_sequences([goods_title_vector_org], maxlen=20, padding='post', truncating='post') #goods_title_vector_cut
        goods_desc = np.hstack((goods_desc_vector, goods_title_vector))
        goods_desc = goods_desc.astype('float32')
        logging.info('goods_desc %s:' % str(goods_desc.shape))
        logging.info(goods_desc)

        goods_image = _url_to_image(mongo_goods['cover'])
        goods_image = goods_image.astype('float32')
        logging.info('goods_image %s:' % str(goods_image.shape))
        logging.info(goods_image)
        return goods_info, goods_desc, goods_image
    else:
        raise UserWarning('cant find goods: %s' % goods)

def get_activity_users(no_order=False):
    user_ids = set()

    connection = _connect_mysql('super_mammy')#pymysql.connect(host='10.66.131.23', user='root', password='yiya1504!!', db='super_mammy')
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `id` FROM `user` WHERE `last_login_time`>%s"
            before7day = time.strftime("%Y-%m-%d", time.localtime(time.time()-7*24*3600))
            cursor.execute(sql, (before7day,))
            results = cursor.fetchall()
            user_ids |= set(map(lambda x: int(x[0]), results))
    finally:
        connection.close()
    # print ("count:%d" % len(user_ids))
    logging.info("count:%d", len(user_ids))

    connection = _connect_mysql('super_mammy_shop')
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `user_id` FROM `user_order` WHERE `status`=%s"
            cursor.execute(sql, (2,))
            results = cursor.fetchall()
            order_user_ids = set(map(lambda x: int(x[0]), results))
            user_ids |= order_user_ids
    finally:
        connection.close()
    logging.info("count:%d", len(user_ids))

    es = Elasticsearch(['10.105.78.165:9200','10.105.1.33:9200'])
    res = es.search(index='logstash-action', doc_type='goods-detail', body={"aggs":{"activity_users":{"terms":{"field":"user_id","size":0}}}})
    # user_see_goods = res['aggregations']['activity_users']['buckets']
    user_ids |= set(map(lambda x: int(x['key']), filter(lambda x: x['doc_count'] >= 200, res['aggregations']['activity_users']['buckets'])))
    logging.info("count:%d", len(user_ids))

    if no_order:
        user_ids -= order_user_ids

    return user_ids

channel_dict = {'appstore':1, 'yingyongbao':2, 'wandoujia':3, 'baidu':4, 'huawei':5, 'official':6, 'meizu':7, 'xiaomi':8, 'sanxing':9}
def user_train_data(user_id):
    user_info, collection_goodses = _user_inherent_info(user_id)

    # low_goods_ids = set()
    # middle_goods_ids = set(mongo_user['goodses'])
    # high_goods_ids = set()
    user_view_detail = _user_recent_view(user_id)
    for collect_goods in set(collection_goodses):
        goods_info, goods_desc, goods_image = _goods_train_data(collect_goods)
        yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], 1

    connection = _connect_mysql('super_mammy_shop')
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `goods_id`,`update_time` FROM `shopping_cart` WHERE `user_id`=%s AND `status`=%s ORDER BY `update_time` DESC"
            cursor.execute(sql, (user_id,1))
            results = cursor.fetchall()
            for res in results:
                user_view_detail = _user_recent_view(user_id, res[1])
                goods_info, goods_desc, goods_image = _goods_train_data(res[0])
                yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], 1
                goods_info, goods_desc, goods_image = _no_view_goods(res[1])
                yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], 0

            sql = "SELECT `goods_id`,`create_time` FROM `order_goods` WHERE `user_id`=%s"
            cursor.execute(sql, (user_id,))
            results = cursor.fetchall()
            for res in results:
                user_view_detail = _user_recent_view(user_id, res[1])
                goods_info, goods_desc, goods_image = _goods_train_data(res[0])
                yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], 2
                goods_info, goods_desc, goods_image = _no_view_goods(res[1])
                yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], 0
    finally:
        connection.close()

def user_predict_data(user_id):
    user_info, _ = _user_inherent_info(user_id)
    user_view_detail = _user_recent_view(user_id)

    client = _connect_mongodb()
    mongo_goodses = client.shiji_shop.goods.find({'status':3})
    for mongo_goods in mongo_goodses:
        goods_info, goods_desc, goods_image = _goods_train_data(mongo_goods)
        yield [user_info, user_view_detail, goods_info, goods_desc, goods_image], mongo_goods['_id']

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
    for test_user in get_activity_users():
        for test_data in user_train_data(test_user):
            print test_data
            break
        break
