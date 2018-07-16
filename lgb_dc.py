from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error
from sklearn.model_selection import KFold
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from six.moves import cPickle as pickle
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

resource_path='resource/lgb/'
data_path='data/'

@contextmanager
def timer(name):
    start=time.clock()
    yield
    print(f'[{name}] done in {time.clock() - start:.0f} s')

def get_dataset_x(df:pd.DataFrame)->pd.DataFrame:
    df['text']=df['Discuss'].fillna(' ')
    df['len_text']=df['Discuss'].apply(lambda x:len(x))
    return df[['text','len_text']]

def get_d2v_x(train_docs:list,train_tags:list,test_docs:list,vec_size=200)->list:
    trainX=[]
    for i,text in enumerate(train_docs):
        word_list=' '.join(jieba.cut(text))
        document=TaggedDocument(word_list.split(' '),train_tags[i])
        trainX.append(document)

    if os.path.exists(resource_path+'d2v.model'):
        model_dv=Doc2Vec.load(resource_path+'d2v.model')
    else:
        # train doc2vec
        with timer('train doc2vec'):
            model_dv=Doc2Vec(trainX,min_count=1, window = 3, size = vec_size, sample=1e-3, negative=5, workers=4)
            model_dv.train(trainX, total_examples=model_dv.corpus_count, epochs=70)
            model_dv.save(resource_path+'d2v.model')

    train_vec=[]
    for x in train_docs:
        word_list=' '.join(jieba.cut(x))
        train_vec.append(model_dv.infer_vector(word_list.split(' ')))

    test_vec=[]
    for x in test_docs:
        word_list=' '.join(jieba.cut(x))
        test_vec.append(model_dv.infer_vector(word_list.split(' ')))

    return train_vec,test_vec

def on_field(f:str,*vec)->Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f),validate=False),*vec)

def to_records(df:pd.DataFrame)->List[Dict]:
    return df.to_dict(orient='records')

def model_lgb(trainX,testX,trainY)->np.array:

    trainY=[x-1 for x in trainY]

    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class':5,
        'is_training_metric':True,
        'early_stopping':10,
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
    }

    trainX,validX,trainY,validY=train_test_split(trainX,trainY,test_size=0.2,random_state=1)

    lgb_train=lgb.Dataset(data=trainX,label=trainY)
    lgb_valid=lgb.Dataset(data=validX,label=validY,reference=lgb_train)

    gbm=lgb.train(params=params,train_set=lgb_train,num_boost_round=2300,valid_sets=lgb_valid)
    gbm.save_model(resource_path+'lgb.model')

    pred=gbm.predict(testX,num_iteration=gbm.best_iteration)

    #临时
    with open(resource_path+'result.pik','wb') as f:
        pickle.dump(pred,f)

    pred=[np.argmax(x)+1 for x in pred]

    return pred

def model_svm(trainX,testX,trainY):
    clf=SVC()
    clf.fit(trainX,trainY)
    result=clf.predict(testX)
    return result

def store_result(pred_ls):
    test=pd.read_csv(data_path+'test.csv')
    test['pred']=pred_ls
    test[['Id','pred']].to_csv(resource_path+'result.csv',index=None,header=None)

def main1():
    vectorizer=make_union(
        on_field('text',Tfidf(max_features=300000,token_pattern='\w+',ngram_range=(1,2))),
        on_field(['len_text'],FunctionTransformer(to_records,validate=False),DictVectorizer())
    )

    with timer('process train'):
        if os.path.exists(resource_path+'dataset.pik'):
            with open(resource_path+'dataset.pik','rb') as f:
                trainX,testX,trainY=pickle.load(f)
        else:
            train=pd.read_csv(data_path+'train.csv')
            train['Discuss']=train['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))

            test=pd.read_csv(data_path+'test.csv')
            test['Discuss']=test['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))

            train=train[train['Score']>0].reset_index(drop=True)
            trainY=train['Score'].values
            trainX=vectorizer.fit_transform(get_dataset_x(train)).astype(np.float32)
            testX=vectorizer.fit_transform(get_dataset_x(test)).astype(np.float32)

            sk=SelectKBest(chi2,k=100000)
            trainX=sk.fit_transform(trainX,trainY)
            testX=sk.transform(testX)

            with open(resource_path+'dataset.pik','wb') as f:
                pickle.dump((trainX,testX,trainY),f)

        print(f'trainX: {trainX.shape} of {trainX.dtype} with{type(trainX)}')
        print(f'testX: {testX.shape} of {testX.dtype} with{type(testX)}')

        #pred=model_lgb(trainX,testX,trainY)
        pred=model_svm(trainX,testX,trainY)
        store_result(pred)

def main():
    with timer('process train'):
        train=pd.read_csv(data_path+'train.csv')
        test=pd.read_csv(data_path+'test.csv')

        #将score转化为list的list以供taggeddocment使用
        score_list=train['Score'].values.tolist()
        score_list=[[x] for x in score_list]
        train_vec,test_vec=get_d2v_x(train['Discuss'].values.tolist(),score_list,test['Discuss'].values.tolist())
        trainX=pd.DataFrame(train_vec)
        trainX['len_text']=train['Discuss'].apply(lambda x:len(x))
        trainY=train['Score'].values

        testX=pd.DataFrame(test_vec)
        testX['len_text']=train['Discuss'].apply(lambda x:len(x))

        print(f'trainX: {trainX.shape} with{type(trainX)}')
        print(f'testX: {testX.shape} with{type(testX)}')
        #pred=model_lgb(trainX,testX,trainY)
        pred=model_svm(trainX,testX,trainY)
        store_result(pred)


if __name__ == '__main__':
    # sentence=[b'waste of time.', b'a shit movie.', b'a nb movie.', b'I love this movie!', b'shit.', b'worth my money.', b'sb movie.', b'worth it!']
    # sentence=['他','你我','这']
    # print(get_cv_one_vec(sentence).toarray())

    # sentence=['它会检查你已经拥有的库文件是否有更新的版本。','这个问题真是郁闷了我一天，网上也是各种找解决方案']
    # tf=TfidfVectorizer(token_pattern='\w+',ngram_range=(1,2),binary=True)
    # # print(tf.fit_transform(sentence).toarray())
    # arr=tf.fit_transform(sentence)
    # x=[x.astype(np.bool).astype(np.float32) for x in arr]
    # print(x[1].toarray())
    #
    # with open(resource_path + 'result.pik', 'rb') as f:
    #     pred=pickle.load(f)
    # pred=[np.argmax(x)+1 for x in pred]
    # print(len(pred))
    # store_result(pred)

    # train_vec,test_vec=get_d2v_x(['根据官方api探索性的做了些尝试。','infer新文档向量'],[[0],[1]],['后期会继续改进'])
    # print(train_vec)
    # print(type(train_vec))
    # print(test_vec)

    # main1()
    main()

