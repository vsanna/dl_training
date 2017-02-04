#coding: UTF-8
#============
def td(f_name):
    return np.genfromtxt("%s" % f_name,
                         delimiter="\t",
                         comments=None,
                         dtype=[('DATA1', 'S25'), ('DATA2','S200')])
# comments=#などと設定しておくと、#以降を無視する
# dtype=[('DATA1', 'S25')] ... DATA1列は文字列で25byteという意味. 列に名前を付与してもる。
# 指定より長いデータは途中で切られて読み込まれる

#===========


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print('Train Start')

# 1. データの読込
f_name = 'data_3000.tsv'
twi_data = td(f_name)


# 2. データをbowに素性選択(特徴抽出)
# bowを記憶するし、ベクトル化する関数
# min_df: 1回しか出ていない単語を足切りしている(max_dfもある)
# token_pattern: ...でも読み込める？
# CountVectorizer#fit(strのarr) ... スペース区切りで覚えてくれる
word_vectorizer = CountVectorizer(min_df=1, token_pattern=u'(?u)\\b\\w+\\b')
word_vectorizer.fit(twi_data['DATA2']) # BoWの生成
# BoWの表示
# print(len(word_vectorizer.get_feature_names())) #=> 279
# for word in word_vectorizer.get_feature_names():
    # print(word)


# 今word_vectorizerはbowを保有している
# それを元に、各twitter textデータをベクトルに変換している
X_train = word_vectorizer.transform(twi_data['DATA2'])
Y_train = twi_data['DATA1']
# print(X_train[1]) #=> (0, 2)    1 ... タプルの頭の0はよくわからないが、X_train[0]はBoWの2番目のデータを1つもつと解釈する


# 3. 学習
# support vector classifier
#   - 分類する直線を引いているところ
#   - ref: support vector regression: 回帰問題
# C: cost ... あとで解説
classifier = svm.SVC(C=1.0, kernel='rbf')
classifier.fit(X_train, Y_train)

print('Train Finished')




# 4. 予測してみる
print('Test Start')

f_name2 = 'data_20.tsv' # 未知データ

twi_data2 = td(f_name2) 

X_test = word_vectorizer.transform(twi_data2['DATA2'])
Y_test = twi_data2['DATA1']
Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)
# [[5 5]
#  [1 9]]
# この読み方はノートにメモした

out_fname = 'out_twi.txt'
f = open(out_fname, 'w')
z = zip(Y_test, Y_pred, twi_data2['DATA2'])
for t_tag, p_tag, twi in z:
    f.write(("%s\t%s\t%s\n") % (t_tag, p_tag, twi))
f.close()



# 5. confusion_matrixの読み方がわかりにくいので、レポートの形にする
target_names = ['negative', 'positive']
print(classification_report(Y_test, Y_pred, target_names=target_names))

#    precision    recall  f1-score   support
#
#    negative       0.83      0.50      0.62        10
#    positive       0.64      0.90      0.75        10
#
# avg / total       0.74      0.70      0.69        20

# recall: 当たった数 / 予測した数 ... negativeだと5 / 6 = 0.83
# precision: あった数 / データの個数 ... negativeだと5 / 10 = 0.5
# f1-score: 1 / (1/P + 1/R): recallとprecisionの調和平均. ... negativeだと 1 /(0.83 + 0.5) = 0.75



# 6. 精度向上


