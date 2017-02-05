import numpy as np

def sen2vec(sen):
    sen = sen.split(" ")
    result = {}
    for word in sen:
        result[word] = 0 if not(word in result) else result[word]
        result[word] += 1 

    return result

def pretty_sen2vec(vec):
    for key, count in vec.items():
        print(key, count)


# v = sen2vec("hoge geho geho gahgah")
# pretty_sen2vec(v)

v1 = sen2vec("A cat sat on the mat")
pretty_sen2vec(v1)

v2 = sen2vec("Cats are sitting on the mat")
pretty_sen2vec(v2)

v1_vec = np.array(list(v1.values()))
v2_vec = np.array(list(v2.values()))

print( np.dot(v1_vec, v2_vec) / (np.linalg.norm(v1_vec) * np.linalg.norm(v2_vec)) )
