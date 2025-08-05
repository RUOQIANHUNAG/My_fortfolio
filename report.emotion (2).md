```python
import re
import requests
import jieba
from snownlp import SnowNLP
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```


```python
url = 'https://api.bilibili.com/x/v1/dm/list.so?oid=833286904'
headers = {
    'cookie': 'buvid3=C01AC0E7-39F3-4B6B-A3D3-3694B4B9DE3F18568infoc; rpdid=|(u~)YYu|m))0J\'uYuuJ|RkYR; blackside_state=0; _uuid=DDC5D475-7FEC-10D5C-F65C-7F108845D9110128540infoc; buvid4=F33B6F33-E729-11A1-361D-0771A6D0F9A530044-022012522-Ed+iBJe/LoAbuLszV4Kbnw%3D%3D; fingerprint3=94c78d98d4109b56446062e8808c1dba; i-wanna-go-back=-1; buvid_fp_plain=undefined; LIVE_BUVID=AUTO1016437080560804; nostalgia_conf=-1; CURRENT_BLACKGAP=0; hit-dyn-v2=1; is-2022-channel=1; fingerprint=75bbe94c3ceebf271b09fbde540f90f3; buvid_fp=75bbe94c3ceebf271b09fbde540f90f3; DedeUserID=641706985; DedeUserID__ckMd5=dd42c5ea734fa0d5; b_ut=5; CURRENT_QUALITY=80; b_nut=100; SESSDATA=cf642170%2C1680496658%2C1ec66%2Aa1; bili_jct=64c027458903950649f8886ed91f8627; sid=5aabkigx; innersign=1; PVID=5; bp_video_offset_641706985=713538997424488400; CURRENT_FNVAL=4048; b_lsid=91EA95E7_183A8306750',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36'
}
```


```python
#response = requests.get(url=url,headers=headers)

```

response = requests.get(url=url,headers=headers)
response.encoding = 'utf-8'
#print(response.text)
content_list = re.findall('<d p=".*?">(.*?)</d>',response.text)
#print(content_list)
for index in content_list:
    print(index)
    with open ('A:/shixi/danmuiphone.txt',mode = 'a',encoding = 'utf-8') as f:
        f.write(index)
        f.write('\n')
    


```python
type(content_list)

```




    list




```python
resulttext[:2]
```




    ['负\t提升\t哈哈哈哈\t哈哈哈', '红魔\t是\t游戏\t手机\t啊\t大哥\t们\t，\t不是\t一条\t赛道\t的']




```python
text[7]
```




    '下次一定'




```python
#哈工大停用词
stopwords = []
with open("A:/shixi/hagongda.txt",'r', encoding='utf-8') as f2:
    for i in f2:
        i = i.rstrip('\n')
        stopwords.append(i)
        
resulttext_ = []         
for word in resulttext:
    words = word.split('\t')
    #words = [s for s in words if len(s) > 1]
    word = [i for i in words if i not in stopwords]
    if len(word) >0:
        resulttext_.append(word)
    

```


```python
resulttext_[:5]
```




    [['负', '提升'],
     ['红魔', '游戏', '手机', '大哥', '不是', '一条', '赛道'],
     ['为啥', '麒麟', '9000', '没用', '散热', '背', '夹', '跑'],
     ['夭寿', '苹果', '做', '散热'],
     ['导游', '屏幕', '主板', '之间', '块', '铁板']]




```python
result_final = []
for s in resulttext_:
    result_final.append('\t'.join(s))
#result_final[:5]    
```


```python
#通过测试发现使用jieba分词效果更好
s = SnowNLP(result_final[7])
s.sentiments

#s = SnowNLP(text[7])
#s.sentiments
```




    0.5689851767388825




```python
#计算得分，添加score列表
score = []
tag = []
ls_pos = []
ls_neg = []
for sent in result_final:
    s = SnowNLP(sent).sentiments
    score.append(s)
    if s>0.5:
        tag.append(1)
        ls_pos.append(sent)
    else:
        tag.append(0)
        ls_neg.append(sent)
    
        
```


```python
score2 = []
for sent in text:
    s = SnowNLP(sent).sentiments
    score2.append(round(s,2))
score2[:5]
```




    [0.96, 0.98, 0.08, 0.55, 0.03]




```python
#result_final[:10]
```




    ['负\t提升',
     '红魔\t游戏\t手机\t大哥\t不是\t一条\t赛道',
     '为啥\t麒麟\t9000\t没用\t散热\t背\t夹\t跑',
     '夭寿\t苹果\t做\t散热',
     '导游\t屏幕\t主板\t之间\t块\t铁板',
     '下次\t一定',
     '说\t13\t香\t人',
     '下次\t一定',
     '寄\tPU',
     '再\t不\t改\t后面\t可能\t会\t更\t大功率\t芯片\t没法用']




```python
#score[:10]
```




    [0.37973102383409907,
     0.8879111932633647,
     0.06707537647011586,
     0.8399725296724299,
     0.03270929849229398,
     0.5689851767388825,
     0.4421193908000549,
     0.5689851767388825,
     0.21058165908252657,
     0.03866720569928106]




```python
sum(tag)
```




    1953




```python
#统计积极消极的个数
count_pos = sum(tag)
count_neg = len(tag)-sum(tag)
count_neg             

```




    1557




```python
#打包成数据框
emotion = pd.DataFrame()
emotion['comment'] = result_final
emotion['score'] = score
emotion['tag'] = tag
#emotion_pos = emotion[emotion['tag']==1]
#emotion_neg = emotion[emotion['tag']==0]    
```


```python
#emotion.shape
emotion.head(10)
#emotion_pos.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment</th>
      <th>score</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>负\t提升</td>
      <td>0.379731</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>红魔\t游戏\t手机\t大哥\t不是\t一条\t赛道</td>
      <td>0.887911</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>为啥\t麒麟\t9000\t没用\t散热\t背\t夹\t跑</td>
      <td>0.067075</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>夭寿\t苹果\t做\t散热</td>
      <td>0.839973</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>导游\t屏幕\t主板\t之间\t块\t铁板</td>
      <td>0.032709</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>下次\t一定</td>
      <td>0.568985</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>说\t13\t香\t人</td>
      <td>0.442119</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>下次\t一定</td>
      <td>0.568985</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>寄\tPU</td>
      <td>0.210582</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>再\t不\t改\t后面\t可能\t会\t更\t大功率\t芯片\t没法用</td>
      <td>0.038667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>下次\t一定</td>
      <td>0.568985</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>天玑\t8000\t没想到</td>
      <td>0.360898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>众所周知\t原神\t测试\t手机\t性能\t软件</td>
      <td>0.991239</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>库克\t涨停</td>
      <td>0.566132</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>iPhone14\t系列\t新机</td>
      <td>0.673354</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>下次\t一定</td>
      <td>0.568985</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>乔布斯\t库克\t搞\t飞机</td>
      <td>0.139244</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>三连\t后续</td>
      <td>0.513849</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>散热\t背\t夹\t测试\t发热\t高</td>
      <td>0.799051</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>安卓</td>
      <td>0.954560</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#统计词频--计算权重
#jieba.analyse.extract_tags()提取关键字的原理是使用TF-IDF算法
import jieba.analyse as analyse
comment = " ".join(text) #将含有字符串的列表转化为空格为间隔的字符串
jieba.analyse.set_stop_words('A:/shixi/hagongda.txt')
top = jieba.analyse.extract_tags(comment, withWeight = True, topK = 20 )
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>0.209382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>散热</td>
      <td>0.155081</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>0.152919</td>
    </tr>
    <tr>
      <th>3</th>
      <td>苹果</td>
      <td>0.112683</td>
    </tr>
    <tr>
      <th>4</th>
      <td>原神</td>
      <td>0.098809</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14plus</td>
      <td>0.096457</td>
    </tr>
    <tr>
      <th>6</th>
      <td>安卓</td>
      <td>0.087046</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A15</td>
      <td>0.075283</td>
    </tr>
    <tr>
      <th>8</th>
      <td>牙膏</td>
      <td>0.074260</td>
    </tr>
    <tr>
      <th>9</th>
      <td>挤牙膏</td>
      <td>0.073279</td>
    </tr>
    <tr>
      <th>10</th>
      <td>红魔</td>
      <td>0.071124</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13pro</td>
      <td>0.067049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A16</td>
      <td>0.064697</td>
    </tr>
    <tr>
      <th>13</th>
      <td>高通</td>
      <td>0.062251</td>
    </tr>
    <tr>
      <th>14</th>
      <td>手机</td>
      <td>0.060728</td>
    </tr>
    <tr>
      <th>15</th>
      <td>情商</td>
      <td>0.055633</td>
    </tr>
    <tr>
      <th>16</th>
      <td>15</td>
      <td>0.052934</td>
    </tr>
    <tr>
      <th>17</th>
      <td>doge</td>
      <td>0.049405</td>
    </tr>
    <tr>
      <th>18</th>
      <td>烧屏</td>
      <td>0.048228</td>
    </tr>
    <tr>
      <th>19</th>
      <td>骁龙</td>
      <td>0.045876</td>
    </tr>
  </tbody>
</table>
</div>




```python
#import wordcloud
from wordcloud import WordCloud,STOPWORDS
stopwords = STOPWORDS.add('都')
w = WordCloud( width = 1000, height = 700,stopwords=stopwords,background_color = "white",font_path = "msyh.ttc" )
w.generate(''.join (result_final))
w.to_file("result_final.png")
```




    <wordcloud.wordcloud.WordCloud at 0x16c4ea05eb0>




```python
# 词云

w1 = WordCloud( width = 1000, height = 700,stopwords=stopwords,background_color = "white",font_path = "msyh.ttc" )
w1.generate(''.join (ls_pos))
w1.to_file("pos.png")

```




    <wordcloud.wordcloud.WordCloud at 0x16c4ea05ee0>




```python
w2 = WordCloud( width = 1000, height = 700,stopwords=stopwords,background_color = "white",font_path = "msyh.ttc" )
w2.generate(''.join (ls_neg))
w2.to_file("neg.png")
```




    <wordcloud.wordcloud.WordCloud at 0x16c4e3c4a30>




```python
#文本特征抽取tfidf
vectorizer = TfidfVectorizer() #实例化一个转换器对象
wordmtx=vectorizer.fit_transform(result_final)
#print(vectorizer.vocabulary_)  #结果即每个词及其对应的编号
#print(wordmtx)#e.g:（0,9061） 0.19821… 其中0代表第一行文本，9061表示编号为9061的词， 0.19821…表示其权重值
```


```python
from sklearn.model_selection import train_test_split,cross_val_score
x_train, x_test, y_train, y_test = train_test_split(wordmtx, tag, test_size=0.3, random_state=11)
```


```python
#逻辑回归 f1 准确率 召回率 精确率
logitmodel=LogisticRegression()#定义回归模型
logitmodel.fit(x_train,y_train)#训练模型
print(classification_report(y_test,logitmodel.predict(x_test))) #f1得分
```

                  precision    recall  f1-score   support
    
               0       0.81      0.70      0.75       492
               1       0.76      0.86      0.80       542
    
        accuracy                           0.78      1034
       macro avg       0.78      0.78      0.78      1034
    weighted avg       0.78      0.78      0.78      1034
    
    


```python
#from sklearn import datasets, svm, metrics
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
print(classification_report(y_test,knn_clf.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               0       0.55      0.96      0.70       492
               1       0.88      0.30      0.45       542
    
        accuracy                           0.61      1034
       macro avg       0.72      0.63      0.58      1034
    weighted avg       0.73      0.61      0.57      1034
    
    


```python
# 朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
bayes = MultinomialNB()
bayes.fit(x_train,y_train)
print(classification_report(y_test,bayes.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.67      0.75       492
               1       0.75      0.88      0.81       542
    
        accuracy                           0.78      1034
       macro avg       0.79      0.78      0.78      1034
    weighted avg       0.79      0.78      0.78      1034
    
    


```python
## SVM 支持向量机
# 定义分类器，clf 意为 classifier，是分类器的传统命名
from sklearn import svm
clf = svm.SVC(kernel = 'linear')  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
# 训练分类器
clf.fit(x_train,y_train)  # 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 cls 里）
print(classification_report(y_test,clf.predict(x_test)))
```

                  precision    recall  f1-score   support
    
               0       0.89      0.93      0.91       276
               1       0.92      0.88      0.90       269
    
        accuracy                           0.90       545
       macro avg       0.91      0.90      0.90       545
    weighted avg       0.91      0.90      0.90       545
    
    


```python

```
