from django.shortcuts import render
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import nltk
from nltk.corpus import stopwords
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def index(request):
    score = 0
    type = ''
    color = ''
    if request.method == 'POST':
        # Получаем строку
        text = request.POST.__getitem__('name')

        #Обрабатываем её

        # Cтоп-слова
        stopWords = set(stopwords.words('english'))
        stopWords.discard('not')
        stopWords.discard('no')


        # preproc_spacy
        nlp = spacy.load("en_core_web_sm")
        spacy_result = nlp(text)
        new_text = ' '.join([tok.lemma_ for tok in spacy_result if tok.lemma_ not in stopWords])

        #Токенизация
        json_file = 'tokenizer.json'
        with open(json_file, 'r') as f:
            tokenizer = tokenizer_from_json(f.read())

        sequences = tokenizer.texts_to_sequences([new_text])
        test = pad_sequences(sequences, maxlen=300)


        with open('model_1.json', 'r') as f:
            model_1 = model_from_json(f.read())

        with open('model_21.json', 'r') as f:
            model_21 = model_from_json(f.read())

        with open('model_22.json', 'r') as f:
            model_22 = model_from_json(f.read())

        with open('model_31.json', 'r') as f:
            model_31 = model_from_json(f.read())

        with open('model_32.json', 'r') as f:
            model_32 = model_from_json(f.read())

        with open('model_33.json', 'r') as f:
            model_33 = model_from_json(f.read())

        with open('model_34.json', 'r') as f:
            model_34 = model_from_json(f.read())


        model_1.load_weights('best_model_1.h5')
        model_21.load_weights('best_model_21.h5')
        model_22.load_weights('best_model_22.h5')
        model_31.load_weights('best_model_31.h5')
        model_32.load_weights('best_model_32.h5')
        model_33.load_weights('best_model_33.h5')
        model_34.load_weights('best_model_34.h5')

        y1 = model_1.predict(test)
        if (y1<=0.5):
            y2 = model_21.predict(test)
            if (y2<=0.5):
                y3 = model_31.predict(test)
                if (y3 <= 0.5):
                    score = 1
                else:
                    score = 2
            else:
                y3 = model_32.predict(test)
                if (y3 <= 0.5):
                    score = 3
                else:
                    score = 4
        else:
            y2 = model_22.predict(test)
            if (y2 <= 0.5):
                y3 = model_33.predict(test)
                if (y3 <= 0.5):
                    score = 7
                else:
                    score = 8
            else:
                y3 = model_34.predict(test)
                if (y3 <= 0.5):
                    score = 9
                else:
                    score = 10

        if (score <5):
            type = 'negative'
            color = 'style=color:red'
        else:
            type = 'positive'
            color = 'style=color:green'

    #score = 10
    #type = 'positive'
    return render(request, 'main/index.html', {'score': score, 'type': type, 'color': color})
