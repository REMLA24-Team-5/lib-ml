from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Preprocessing:
    """
    Class to preprocess URL datasets
    """
    def __init__(self, x_train, y_train, x_test, x_val):

        # We mark this line as nosec to let Bandit know that this token is not a password token
        self.tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')  # nosec
        self.tokenizer.fit_on_texts(x_train + x_val + x_test)
        self.sequence_length = 200
        self.encoder = LabelEncoder()
        self.encoder = self.encoder.fit(y_train)
    def process_dataset(self, dataset):
        return pad_sequences(self.tokenizer.texts_to_sequences(dataset), maxlen=self.sequence_length)
    def process_URL(self, url):
        return pad_sequences(self.tokenizer.texts_to_sequences([url]), maxlen=self.sequence_length)[0]
    def process_labels(self, labels):
        return self.encoder.transform(labels)
    def process_label(self, label):
        return self.encoder.transform([label])[0]
    def get_char_index(self):
        return self.tokenizer.word_index