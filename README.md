# Emojify
A GRU for text classification, on entering a sentence it can classify the sentence to assign it one of 5 emojis.  
The model is a GRU with 256 hidden units, the final hiddent state is passed to a dense layer with 5 units softmax activation  
300 dimensional pre trained word2vec word vectors are used.
