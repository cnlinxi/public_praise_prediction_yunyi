import os
os.environ['OMP_NUM_THREADS'] = '4'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import tensorflow as tf
import keras as ks
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import math
import jieba
from snownlp import SnowNLP
from snownlp import sentiment
import multiprocessing as mp
from six.moves import cPickle as pickle
from sklearn.metrics import mean_squared_error

data_path='data/'
resource_path='resource/'

#一共原始的特征
@contextmanager  #创建一个上下文管理器，显示运行的时间
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def preprocess(df: pd.DataFrame) -> pd.DataFrame: #标注df是pd.DataFrame类型的，返回的也是pd.DataFrame的
    df['text']=df['Discuss'].fillna('')
    return df[['text','len_discuss']]

def on_field(f: str, *vec) -> Pipeline:# *代表可变长度的
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    #将df由{'a':[1,2,3],'b':[4,5,6]}转换为[{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
    return df.to_dict(orient='records')

def get_nouns(sentences:list)->set:
    noun_list=[]
    for s in sentences:
        # print('sentence',s)
        st=SnowNLP(s)
        tags=list(st.tags)
        # print('tags',tags)
        for index,word in enumerate(st.words):
            if tags[index][1]=='n':
                noun_list.append(word)
                # print('word',word)
    return set(noun_list)

def get_keywords(sentences:list)->set:
    keywords_list=[]
    for s in sentences:
        st=SnowNLP(s)
        keywords_list.extend(st.keywords(limit=3))
    return set(keywords_list)

# def get_noun(sentence):
#     s=SnowNLP(sentence)
#     noun_word_list=[]
#     for index,word in enumerate(s.words):
#         if s.tags[index]=='n':
#             noun_word_list.append(word)
#     return noun_word_list

def filter_noun(nouns:list):
    nouns_filtered=[]
    for x in nouns:
        s=SnowNLP(x)
        if s.sentiments>=0.7 or s.sentences<=0.3:
            nouns_filtered.append(x)
    return nouns_filtered

def seg_sentence(sentence)->list:
    s=SnowNLP(sentence)
    return s.words

def fit_predict(xs, y_train) -> np.ndarray:
    #[ [[Xb_train, Xb_valid], [X_train, X_valid]] ,[[Xb_train, Xb_valid], [X_train, X_valid]] ]
    X_train, X_test = xs
    print("X_train:",X_train.shape)
    print("X_test:",X_test.shape)
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def main(evalation=True):
    #make_union将各个特征组合到一起
    vectorizer = make_union(
        #先获取pd中name，进行Tfidf,根据语料库的出现词的频率排序，选择前300000个词，\w+匹配数字字母下划线的多个字符
        #on_field('name', Tfidf(max_features=1000, token_pattern='\w+')),
        #获取pd中的text，也是tfidf，不同的是使用ngram
        on_field('text', Tfidf(max_features=300000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['len_discuss'],FunctionTransformer(to_records,validate=False),DictVectorizer()),
        #on_field(['shipping', 'item_condition_id'],
                 #FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=1)
    y_scaler = StandardScaler()
    with timer('process train'):
        train = pd.read_csv(data_path+'train_split.csv')
        train['len_discuss']=train['Discuss'].apply(lambda x:len(x))

        train['Discuss']=train['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))

        test=pd.read_csv(data_path+"dev_split.csv")
        test['len_discuss']=test['Discuss'].apply(lambda x:len(x))
        test['Discuss']=test['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))
        y_true=None
        if evalation:
            y_true=test['Score'].values

##################### noun
        # print('load noun set...')
        #
        # if os.path.exists(resource_path+'noun_set.pik'):
        #     with open(resource_path+'noun_set.pik','rb') as f:
        #         noun_set=pickle.load(f)
        #         # noun_set=filter_noun(noun_set)
        # else:
        #     noun_set=get_nouns(train['Discuss'].values)
        #     with open(resource_path+'noun_set.pik','wb') as f:
        #         pickle.dump(noun_set,f)
        #     # noun_set=filter_noun(noun_set)
        #
        # print(f'noun size:{len(noun_set)}')
#######################

###################### keyword
        print('load keyword set...')

        if os.path.exists(resource_path+'keyword_set.pik'):
            with open(resource_path+'keyword_set.pik','rb') as f:
                keyword_set=pickle.load(f)
        else:
            keyword_set=get_keywords(train['Discuss'].values)
            with open(resource_path+'keyword_set.pik','wb') as f:
                pickle.dump(keyword_set,f)

        print(f'keyword size:{len(keyword_set)}')
######################

        train = train[train['Score'] > 0].reset_index(drop=True)#取出所有价格大于0的数据
        # cv = KFold(n_splits=10, shuffle=True, random_state=42)#20折
        # train_ids, valid_ids = next(cv.split(train))
        # valid=train.iloc[valid_ids]
        # train=train.iloc[train_ids]
        y_train_start=train['Score'].values
        y_train=y_scaler.fit_transform(train['Score'].values.reshape(-1,1))
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        X_test=vectorizer.transform(preprocess(test)).astype(np.float32)

        #y_test=valid['Score']

        sk=SelectKBest(chi2,k=100000)
        X_train=sk.fit_transform(X_train,y_train_start)
        X_test=sk.transform(X_test)

        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        print(f'X_test: {X_test.shape} of {X_test.dtype}')
        # del train
    # with timer('process valid'):
    #     X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    with ThreadPool(processes=6) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_test]]

############################### noun
        # vec=CountVectorizer(binary=True,tokenizer=seg_sentence)
        # vec.fit(noun_set)
        # Xn_train,Xn_valid=[vec.transform(x) for x in [train['Discuss'].values,test['Discuss'].values]]
##################################

############################# keyword
        if os.path.exists(resource_path+'keyword_train.pik'):
            with open(resource_path+'resource_train.pik','rb') as f:
                Xk_train,Xk_valid=pickle.load(f)
        else:
            vec=CountVectorizer(binary=True,tokenizer=seg_sentence)
            vec.fit(keyword_set)
            Xk_train,Xk_valid=[vec.transform(x) for x in [train['Discuss'].values,test['Discuss'].values]]
            with open(resource_path+'resource_train.pik','wb') as f:
                pickle.dump([Xk_train,Xk_valid],f)
#############################

############################

####下面的xn_train,Xn_valid

##############

############# 拼接在内部
        # Xb_a_train=np.concatenate([Xb_train,Xk_train],axis=1)
        # Xb_a_valid=np.concatenate([Xb_valid,Xk_valid],axis=1)
        # X_a_train=np.concatenate([X_train,Xk_train],axis=1)
        # X_a_test=np.concatenate([X_test,Xk_valid],axis=1)
        xs = [[Xb_train, Xb_valid], [X_train, X_test],[Xk_train,Xk_valid]]*2 #复制一遍  #Xb表示单词的出现与否，而X使用的是tfidf特征权重
############## 放在训练
        # xs = [[Xb_train, Xb_valid],[X_train, X_test],[Xk_train,Xk_valid]]*2 #复制一遍  #Xb表示单词的出现与否，而X使用的是tfidf特征权重

###############

        print(len(xs),len(xs[0]))
        #print(len(xs[1]))
        xx=pool.map(partial(fit_predict, y_train=y_train), xs)#np.mean指传入多次进行平均
        print(len(xx))
        y_pred = np.mean(xx,axis=0)
        y_pred=y_scaler.inverse_transform(y_pred)
    # print(y_pred)

    pre=[]
    for i in y_pred:
        if i>4.7:
            pre.append(5)
        else:
            pre.append(i)

    if evalation and y_true is not None:
        print('the score is :',evaluate(y_true,pre))

    result=pd.DataFrame({'ID':test.Id,'Discuss':test.Discuss,'Score':pre})
    result.to_csv('MLP_simple_jieba_stopword_chibest.csv',header=None,index=None)

def evaluate(y_true,y_prediction):
    return 1.0/(1.0+np.sqrt((mean_squared_error(y_pred=y_prediction,y_true=y_true))))

if __name__ == '__main__':
    main(evalation=True)