from tensorflow import keras
import io
import numpy as np
import pandas as pd


def load_v(n):      #The function to load the word vectors
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
            yield data

    gen = load_vectors("wiki-news-300d-1M.vec")     #The word vectors must be downloaded and saved here
    data = {}
    for i in range(n):
        data = gen.__next__()
    return data


def gen(word_to_vec):
    df = pd.read_csv("train_emoji.csv")
    X = np.array(df.iloc[:, 0])
    y = np.array(df.iloc[:, 1])
    while True:
        for sent, code in zip(X, y):
            sent = sent.strip()
            words = sent.split()
            tX = np.zeros((1, len(words), 300))
            for i, word in enumerate(words):
                if word in word_to_vec:
                    tX[0, i] = word_to_vec[word]
            ty = np.zeros((1, 5))
            ty[0, code] = 1
            yield (tX, ty)


if __name__ == '__main__':
    word_to_vec = load_v(150000)
    code_to_emoji = {
        0: "❤️",
        1: "⚾",
        2: "☺️",
        3: "😞",
        4: "🍴"
    }
    model = keras.Sequential((
        keras.layers.GRU(256, return_sequences=False, input_shape=(None, 300), batch_size=1),
        keras.layers.Dense(len(code_to_emoji), activation="softmax")))

    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy")
    model.fit_generator(gen(word_to_vec), steps_per_epoch=134, epochs=20)

    while True:
        sent = input()
        sent = sent.strip()
        words = sent.split()
        X = np.zeros((1, len(words), 300))
        try:
            for i, word in enumerate(words):
                if word not in word_to_vec:
                    raise KeyError
                X[0, i] = word_to_vec[word]
            p = list(np.squeeze(model.predict(X, batch_size=1)))
            print(p)
            print(code_to_emoji[p.index(max(p))])
        except KeyError as e:
            print("invalid word")
            continue
