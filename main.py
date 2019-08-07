import yaml
from os import listdir
import nltk
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random

PATH = "data/english/"


class Corpus:
    def __init__(self, d):
        self.categories = None
        self.conversations = None
        for k, v in d.items():
            setattr(self, k, v)


data = []
for fn in listdir(PATH):
    with open(PATH + fn, 'r') as s:
        data.append(Corpus(yaml.safe_load(s)))

stemmer = LancasterStemmer()

clear_sentence = lambda sentence: ' '.join([stemmer.stem(w) for w in nltk.word_tokenize(sentence)])

questions = []
classes = []

for item in data:
    cat = item.categories[0]
    for quest in item.conversations:
        for q in quest:
            questions.append(clear_sentence(q))
            classes.append(cat)

tfv = TfidfVectorizer(stop_words='english')
le = LabelEncoder()

X = tfv.fit_transform(questions)
y = le.fit_transform(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = SVC(kernel='linear')
model.fit(X, y)
print("SVC =", model.score(X_test, y_test))


def response(text):
    t_text = tfv.transform([clear_sentence(text.strip().lower())])
    class_ = le.inverse_transform(model.predict(t_text))
    for d in filter(lambda cls: cls.categories[0] in class_, data):
        cos_sims = []
        conversions = []
        for conversion in d.conversations:
            sims = cosine_similarity(tfv.transform(conversion), t_text)
            for i, cs in enumerate(sims):
                cos_sims.append(cs)
                conversions.append(conversion)
        print("Bot :", random.choice(conversions[cos_sims.index(max(cos_sims))][1:]))


while True:
    response(input("You : "))
