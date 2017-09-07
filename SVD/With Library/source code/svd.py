from nltk import FreqDist
from operator import itemgetter
import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
sys.stdout=open('log.txt','w')
plt.ion()

#reading given corpus.
words=[]
with open('text8','r') as f:
    for line in f:
        for word in line.split():
           words.append(word) 
f.close()

ABSS_word=words[:100000]

#finding most 10000 frequent words from given corpus.
f=open('Most_Fre_data.txt','w')
f.write('Words frequency in given corpus')
f.write('\n')
fdist = FreqDist(ABSS_word)
dict1=fdist.most_common(1000)
dict1=dict(dict1)

sorted_dict = sorted(dict1.items(), key=itemgetter(1), reverse=True)
lis=[item[0] for item in sorted_dict]
lis.append("UNK")

#string most 10000 frequent word with their frequency.
for item in sorted_dict:
   f.write(str(item[0]))
   f.write(' = ')
   f.write(str(item[1]))
   f.write('\n')
f.close()

#replacing all non 10000 frequent words to UNK and making new corpus.
f=open('Modified_corpus.txt','w')

for index, item in enumerate(ABSS_word):
  if item not in dict1:
      ABSS_word[index] = "UNK"

  f.write(ABSS_word[index])
  f.write(" ")
f.close()

ABSS_word2=[]
with open('Modified_corpus.txt','r') as f:
    for line in f:
        for word in line.split():
          ABSS_word2.append(word) 
f.close()


#creating a dataframe, row and col both are unqiue words from modified corpus.
x= lis
y = lis

co_out=pd.DataFrame(index=x,columns=y)

for i in range(0,co_out.shape[0]):
    for word in y:
        co_out.ix[i,str(word)]= 0


#co occurrence matrix with window size 5.
length=len(ABSS_word2)
window =5
for i, item in enumerate(ABSS_word2):
  for j in range(max(i-window,0),min(i+window+1,length)):
     co_out.ix[item, ABSS_word2[j]]+=1

for i, item in enumerate(ABSS_word2):
    co_out.ix[item,item]-=1
print "Co-occurrence Matrix: "
print (co_out)
sys.stdout.close()

sys.stdout=open('U_Sigma_V.txt','w')

#converting dataframe to nd array.
co_out=co_out.as_matrix()

#converting ndarray to matrix.
co_out=np.asmatrix(co_out)

#converting matrix to list
co_out=co_out.tolist()

#SVD decompose co occurrence  matrix into U, s, V.
U, s, V = linalg.svd( co_out, full_matrices=False)
print "U: ", U.shape
print U
print "\n\n"
print "sigma:", s.shape
print s
print s[:50]
print "\n\n"
print "VT:", V.shape
print V 
print "\n\n"
sys.stdout.close()

sys.stdout=open('vector_embedding.txt','w')
#Obtaining 50 dimensional vector embedding for each word.
U=U[0:100,:50]
print U
sys.stdout.close()

#to plot vector embedding in 2 dimension
tsne = TSNE(n_components=2, random_state=0)

final_matrix=tsne.fit_transform(U[:,:])

words = [lis[i] for i in range(0,100)]

fig = plt.figure(figsize=(20,20))
for i, label in enumerate(words):
    x, y = final_matrix[i,:]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
plt.title('Vector Embedding using SVD from library')
plt.ylabel('X')
plt.xlabel('Y')
plt.show(block=True)







