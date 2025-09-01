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
from keras.layers import LSTM, Dropout, Dense, SpatialDropout1D, Embedding
from sklearn.model_selection import train_test_split

# Global constants
cnt_tr = 0
MAX_NB_WORDS = 50000      # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 250 # Max length of input sequences
EMBEDDING_DIM = 100       # Embedding vector dimension

# Regex patterns for cleaning text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean the input text:
    - Lowercase
    - Replace certain symbols with space
    - Remove bad symbols
    - Remove stopwords
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')   # custom cleaning
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

# Load dataset
df = pd.read_csv('K:\\Chunk2\\test22.csv')

# Keep only necessary columns
for i in df:
    if i != 'Product' and i != 'Consumer complaint narrative':
        df.drop(i, inplace=True, axis=1)

# Normalize product categories (grouping similar ones together)
df.loc[df['Product'] == 'Credit reporting', 'Product'] = 'Credit reporting, credit repair services, or other personal consumer reports'
df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit card or prepaid card'
df.loc[df['Product'] == 'Prepaid card', 'Product'] = 'Credit card or prepaid card'
df.loc[df['Product'] == 'Payday loan', 'Product'] = 'Payday loan, title loan, or personal loan'
df.loc[df['Product'] == 'Virtual currency', 'Product'] = 'Money transfer, virtual currency, or money service'
df.loc[df['Product'] == 'Money transfers', 'Product'] = 'Money transfer, virtual currency, or money service'
df.loc[df['Product'] == 'Student loan', 'Product'] = 'Consumer Loan'
df.loc[df['Product'] == 'Vehicle loan or lease', 'Product'] = 'Consumer Loan'
df.loc[df['Product'] == 'Payday loan, title loan, or personal loan', 'Product'] = 'Consumer Loan'
df.loc[df['Product'] == 'Checking or savings account', 'Product'] = 'Bank account or service'
df.loc[df['Product'] == 'Money transfer, virtual currency, or money service', 'Product'] = 'Bank account or service'

# Remove underrepresented categories
df = df[df.Product != 'Other financial service']
df = df[df.Product != 'Bank account or service']
df = df[df.Product != 'Consumer Loan']
df = df[df.Product != 'Credit card or prepaid card']

# Reset index after filtering
df = df.reset_index(drop=True)

# Clean complaint narratives
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(clean_text)
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.replace('\d+', '') # remove digits

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, 
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
                      lower=True)
tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df['Consumer complaint narrative'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# One-hot encode labels
Y = pd.get_dummies(df['Product']).values

# Split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def graph_plots(history, string):
    """Helper function to plot training vs validation metrics"""
    plt.title(string + ' vs val_' + string)
    plt.plot(history.history[string], 'r')
    plt.plot(history.history['val_' + string], 'b')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def funcLSTM(dropout_array=[]):
    """
    Build and train an LSTM model.
    dropout_array: [dropout_after_first_LSTM, dropout_after_second_LSTM]
    """
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(dropout_array[0]))
    model.add(LSTM(100))
    model.add(Dropout(dropout_array[1]))
    model.add(Dense(3, activation='softmax'))  # 3-class classification

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 10
    batch_size = 256

    # Early stopping to prevent overfitting
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
    )

    # Evaluate model on test set
    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.7f}\n  Accuracy: {:0.7f}'.format(accr[0], accr[1]))

    # Plot training curves
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

def PSO():
    """
    Particle Swarm Optimization (PSO) to optimize dropout rates for the LSTM model.
    Each particle represents [dropout1, dropout2].
    """
    w = 0.5   # inertia weight
    c1 = 0.5  # cognitive parameter
    c2 = 0.9  # social parameter
    target = 1

    n_it = 5   # number of iterations
    error = 1e-5
    n_pr = 10  # number of particles

    # Initialize particle positions randomly in [0,1] range
    particle_pos_vec = numpy.random.rand(n_pr, 2)
    pbest_pos = particle_pos_vec
    pbest_fitness_val = numpy.array([0.0 for _ in range(n_pr)])
    gbest_pos = numpy.array([float('inf'), float('inf')])
    gbest_fitness_val = 0.0

    velocity_vec = ([numpy.array([0, 0]) for _ in range(n_pr)])
    it = 0

    # Main PSO loop
    while it < n_it and abs(gbest_fitness_val - target) > error:
        for i in range(n_pr):
            global cnt_tr
            cnt_tr += 1
            print(f"Iteration {it}, Particle {i}, Training count: {cnt_tr}")
            print("Current position:", particle_pos_vec[i])

            # Evaluate particle fitness
            fitness = funcLSTM(particle_pos_vec[i])

            # Update personal best
            if fitness > pbest_fitness_val[i]:
                pbest_fitness_val[i] = fitness
                pbest_pos[i] = particle_pos_vec[i]

            # Update global best
            if fitness > gbest_fitness_val:
                gbest_fitness_val = fitness
                gbest_pos = particle_pos_vec[i]
                print("Updated Global Best!")

        # Update velocity and position of each particle
        for i in range(n_pr):
            new_velocity = (w * velocity_vec[i]) \
                           + (c1 * random.random()) * (pbest_pos[i] - particle_pos_vec[i]) \
                           + (c2 * random.random()) * (gbest_pos - particle_pos_vec[i])
            new_pos = new_velocity + particle_pos_vec[i]
            particle_pos_vec[i] = new_pos

        it += 1

    print("\nFinal Global Best Position:", gbest_pos)
    print(f"Best Fitness Value: {gbest_fitness_val} after {it} iterations")

# Run PSO optimization
PSO()
