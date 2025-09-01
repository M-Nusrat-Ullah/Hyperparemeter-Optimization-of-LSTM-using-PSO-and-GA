import re
import numpy
import random
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from plotly.offline import iplot
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout, Dense, Embedding
from sklearn.model_selection import train_test_split

# Global counters and GA parameters
cnt_tr = 0             # Training counter (keeps track of how many times model is trained)
n_it = 5               # Number of generations
n_bits = 16            # Number of bits used to encode a parameter
n_pop = 10             # Population size
r_cross = 0.9          # Crossover probability
r_mut = 1.0 / (float(n_bits) * 2.0)  # Mutation probability

# LSTM parameters
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

# Regex rules for text cleaning
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# ---------- TEXT CLEANING ----------
def clean_text(text):
    """Clean raw text by removing special characters, stopwords, and digits."""
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')  
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

# ---------- LOAD & PREPROCESS DATA ----------
df = pd.read_csv('K:\\Chunk2\\test22.csv')

# Keep only 'Product' and 'Consumer complaint narrative'
for i in df:
    if i not in ['Product', 'Consumer complaint narrative']:
        df.drop(i, inplace=True, axis=1)

# Normalize product categories into fewer groups
df.loc[df['Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
df.loc[df['Product'].isin(['Credit card', 'Prepaid card']), 'Product'] = 'Credit card or prepaid card'
df.loc[df['Product'].isin(['Payday loan', 'Payday loan, title loan, or personal loan']), 'Product'] = 'Consumer Loan'
df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
df.loc[df['Product'] == 'Money transfers', 'Product'] = 'Money transfer, virtual currency, or money service'
df.loc[df['Product'].isin(['Student loan', 'Vehicle loan or lease']), 'Product'] = 'Consumer Loan'
df.loc[df['Product'] == 'Checking or savings account', 'Product'] = 'Bank account or service'
df.loc[df['Product'] == 'Money transfer, virtual currency, or money service', 'Product'] = 'Bank account or service'

# Remove some categories entirely
df = df[~df['Product'].isin(['Other financial service', 'Bank account or service', 'Consumer Loan', 'Credit card or prepaid card'])]

# Reset index
df = df.reset_index(drop=True)

# Clean text column
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.replace('\d+', '')

# ---------- TOKENIZATION ----------
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
word_index = tokenizer.word_index

# Convert text to padded sequences
X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Convert labels to one-hot encoding
Y = pd.get_dummies(df['Product']).values

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------- TRAINING & EVALUATION ----------
def graph_plots(history, string):
    """Plot accuracy/loss graphs."""
    plt.title(string + ' vs val_' + string)
    plt.plot(history.history[string], 'r')
    plt.plot(history.history['val_' + string], 'b')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def funcLSTM(dropout_array=[]):
    """Build, train, and evaluate an LSTM model with given dropout rates."""
    global cnt_tr
    cnt_tr += 1
    print(f"\nTraining #{cnt_tr} with dropout={dropout_array}")

    # Define LSTM model
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(dropout_array[0]))
    model.add(LSTM(100))
    model.add(Dropout(dropout_array[1]))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    history = model.fit(
        X_train, Y_train, epochs=10, batch_size=256,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
    )

    # Evaluate
    accr = model.evaluate(X_test, Y_test)
    print(f"Test set\n  Loss: {accr[0]:0.7f}\n  Accuracy: {accr[1]:0.7f}")

    # Plot metrics
    plt.title('Loss vs Accuracy')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.plot(history.history['loss'], 'r', label='loss')
    plt.plot(history.history['accuracy'], 'b', label='accuracy')
    plt.legend()
    plt.show()

    graph_plots(history, "accuracy")
    graph_plots(history, "loss")

    return accr[1] * 100

# ---------- GENETIC ALGORITHM ----------
def decode(bitstring):
    """Decode bitstring into dropout values (scaled between 0 and 1)."""
    decoded = []
    high = 2 ** n_bits
    for i in range(2):  # Two dropout values
        start = i * n_bits
        end = start + n_bits
        sub_str = bitstring[start:end]
        number = int(''.join(str(s) for s in sub_str), 2)
        value = number / high
        decoded.append(value)
    return decoded

def selection(pop, fitness):
    """Tournament selection: pick one from random subset of population."""
    ran_pop = numpy.random.randint(len(pop))
    for i in numpy.random.randint(0, len(pop), 2):
        if fitness[i] > fitness[ran_pop]:
            ran_pop = i
    return pop[i]

def crossover(p1, p2):
    """Single-point crossover between two parents."""
    c1, c2 = p1.copy(), p2.copy()
    if numpy.random.rand() < r_cross:
        pt = numpy.random.randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring):
    """Bit-flip mutation."""
    for i in range(len(bitstring)):
        if numpy.random.rand() < r_mut:
            bitstring[i] ^= 1

def genetic_algo():
    """Run genetic algorithm to optimize dropout parameters for LSTM."""
    pop = [numpy.random.randint(0, 2, n_bits * 2).tolist() for _ in range(n_pop)]
    best = 0
    best_fitness = funcLSTM(decode(pop[0]))  # Evaluate first candidate

    for gen in range(n_it):
        decoded = [decode(p) for p in pop]
        fitness = [funcLSTM(v) for v in decoded]

        # Track best solution
        for i in range(n_pop):
            if fitness[i] > best_fitness:
                best, best_fitness = pop[i], fitness[i]
                print(f"Gen {gen}, decode={decoded[i]}, fitness={best_fitness:.4f}")

        # Selection and reproduction
        elite = [selection(pop, fitness) for _ in range(n_pop)]
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = elite[i], elite[i+1]
            for c in crossover(p1, p2):
                mutation(c)
                children.append(c)
        pop = children

    return [best, best_fitness]

# Run GA
best, best_fitness = genetic_algo()
print("Best fitness:", best_fitness)
