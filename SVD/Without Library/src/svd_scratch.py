import sys
import cmath
import numpy.linalg
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

words=[]
with open('text8','r') as f:
    for line in f:
        for word in line.split():
           words.append(word) 
f.close()

#take first 100000 words
words1=words[:100000]

#dictionary to keep count of words in reduced corpus
dict1={}
for i,word in enumerate(words1):
	if word not in dict1.keys():
		dict1[word]=1
	else:
		dict1[word]=dict1.get(word, 0) + 1

#sort dictionary based on count
sortdict= sorted(dict1, key=dict1.get, reverse=True)

#take top 1000 words
vocab=sortdict[:1000]

#replace less frequent words in reduced corpus by UNK
for i,word in enumerate(words1):
	if word in vocab:
		continue		
	else:
		words1[i]="UNK"

#append UNK into vocabulary
vocab.append("UNK")

#create a dictionary of above vocabulary to map words in this vocabulary to their respective index
vocabulary={}
for i,word in enumerate(vocab):
	vocabulary[word]=i

#save this vocabulary into a dictionary for this reduced corpus
with open('dictionary','wb') as f1:
	pickle.dump(vocabulary,f1)
with open('dictionary','rb') as f1:
	vocab2=pickle.load(f1)

#co-occur matrix
matrix = [[0 for x in range(len(vocab))] for y in range(len(vocab))] 

#function to update co-occur matrix
def updatematrix(row,col):
	matrix[row][col]+=1
	return 

#function to find index of word from vocabulary
def findindex(word):
	return vocab.index(word)

#window size=5
l=5
for i, target in enumerate(words1):
	row=findindex(target)
	for j in range(max(i-l,0),min(i+l+1,len(words1))):
		if(i==j):	
			continue
		else:
			context=words1[j]
			col=findindex(context)
			updatematrix(row,col)

#calculate A=matrix*matrixtranspose(since here it is symmetric matrix so can use matrix itself)
A=[[0 for x in range(len(vocab))] for y in range(len(vocab))] 
for i in range(len(vocab)):
	for j in range(len(vocab)):
		A[i][j]+=matrix[i][j]*matrix[i][j]

#eignevalues and eigenvectors of A
Evalues,Evectors=numpy.linalg.eig(A)

#create a dictionary of eigen values to store their indices from list of eigen values
sigmadict={}
j=0

for i in Evalues:
	if i in sigmadict.keys():
		print "error"
		break
	else:
		sigmadict[i]=j
		j=j+1

#sort the dictionary of index of eigen values 
sorteddict= sorted(sigmadict.items(),reverse=True)

#create a list of sorted eigen values
sortedEvalues=[item[0] for item in sorteddict]

#make a new matrix of eigen vectors based on decreasing eigen values
sortedEvectors=[[0 for x in range(len(vocab))] for y in range(len(vocab))] 
k=0
for i,tup in enumerate(sorteddict):		
	ind=tup[1]					#get the index of eigen value in original list from the dictionary of eigen values
	if(i==ind):					#eigenvalue is in its own position in the list of sorted eigenvalues
		continue
	else:						#eigenvalue has changed its position in the list of sorted eigenvalues
		for j in range(len(vocab)):
			sortedEvectors[j][k]=-Evectors[j][ind]
		k=k+1

#truncate the eigenvector matrix upto 50 columns(dimensions)
truncU=[[0 for x in range(50)] for y in range(len(vocab))] 
for i in range(len(sortedEvectors)):
	for j in range(50):
		truncU[i][j]=sortedEvectors[i][j]

#dump this matrix of vector representation of size len_vocab X 50 to truncU 
with open('vector_matrix','wb') as f1:
	pickle.dump(truncU,f1)

#save the vector representation of words from vocabulary
sys.stdout=open('vector_rep','w')
print "vector representation of words in vocabulary\n"
for i in range(len(vocab)):
	print str(vocab[i]).ljust(10),
	for j in range(50):
		print str(truncU[i][j]).ljust(10),
	print "\n"
sys.stdout.close()

#matrix for vector representation of desired 100 words(100X50)
wordvector=[[0 for x in range(50)] for y in range(100)] 
k=0
for i in range(0,100):
	for j in range(50):
		wordvector[k][j]=truncU[i][j]
	k+=1

#to plot vector embedding in 2 dimension redusing it from 50 dimensions
tsne = TSNE(n_components=2, random_state=0)
final_matrix=tsne.fit_transform(wordvector)
hundredwords=vocab[0:100]	#get those words from vocabulary to use them as labels
fig = plt.figure(figsize=(20,20))
for i, label in enumerate(hundredwords):
    x, y = final_matrix[i,:]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
plt.title('Vector Embedding using SVD from scratch implementation')
plt.ylabel('X')
plt.xlabel('Y')
plt.show(block=True)
#fig.savefig('SVDscratch.png', bbox_inches='tight')
