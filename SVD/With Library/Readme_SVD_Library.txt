Readme file for SVD with library implementation.

1. Input: 'text8' corpus should present in same directory.
   We are running the program for first 100000 words.

2. Run svd.py file in terminal:
    - python svd.py

3. output:
   a)'Modified_corpus.txt' contains the most frequent word and not rare words are replaced by UNK.
   b)'Most_Fre_data.txt' contains 1000 most frequent data from 100000 words.
   c)'log.txt' file contains Co-occurrence matrix for 1000 most frequent words.
   d)After applying SVD, co-occurrence matrix is decomposed into U, sigma, V matrices. These are present in 
     'U_Sigma_V.txt'.
   e)50 Dimensional vector representation of 0 to 100 most frequent words is present in 'vector_embedding.txt'.
   f)Plot of these vector representation is present in 'vector_emb_svd_library.png'.

