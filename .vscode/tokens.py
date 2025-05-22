import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from translate import Translator
from textblob import TextBlob

sample_text = 'lowi: fkkk my wet hole daddy pls'
tokens = word_tokenize(sample_text)
vectorizer = CountVectorizer()
translator = Translator(to_lang='es')  # Spanish

print('Tokens:', tokens)

unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print('Unigrams:', unigrams)
print('Bigrams:', bigrams)
print('Trigrams:', trigrams)

texts = [
  'I love programming.', 'Python is amazing.',
  'I enjoy machine learning.', 'The weather is nice today.', 'I like algo.',
  'Machine learning is fascinating.', 'Natural Language Processing is a part of AI.'
]

labels = [
  'tech', 'tech', 'tech', 'non-tech', 'tech', 'tech', 'tech'
]

x = vectorizer.fit_transform(texts)
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

text = 'I love progamming and machine learnig.'
translation = translator.translate(text)
print('Translated Text:', translation)
blob = TextBlob(text)
corrected_text = blob.correct()

# Print the corrected text
print('Original Text:', text)
print('Corrected Text:', corrected_text)