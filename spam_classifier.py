import pandas as pd

data = pd.read_table('SMSSpamCollection', sep="\t", header=None, names=['label', 'sms_msg'])
data.head()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data.iloc[:, 0] = label_encoder.fit_transform(data.iloc[:, 0])

data.isnull().any()
#
# documents = ['Hello Mohit, how are you!',
#              'Win money, win from home.',
#              'Call me now.',
#              'Hello Sneha, Call hello you tomorrow?']
#
# # first lower case the text
# lower_case_letter = []
# for i in documents:
#     lower_case_letter.append(i.lower())
# print(lower_case_letter)
#
# # then remove punctuations
# sans_punctuation = []
#
# import string
#
# for i in lower_case_letter:
#     sans_punctuation.append(i.translate(str.maketrans("", "", string.punctuation)))
# print(sans_punctuation)
#
# # tokenising
# preprocessed_sentence = []
# for i in sans_punctuation:
#     preprocessed_sentence.append(i.split(' '))
# print(preprocessed_sentence)
#
# # count_frequency
# count_frequency = []
# import pprint
# from collections import Counter
#
# for i in preprocessed_sentence:
#     freq = Counter(i)
#     count_frequency.append(freq)
# pprint.pprint(count_frequency)

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
print(count_vector)
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()

feature_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data['sms_msg'], data['label'], test_size=0.2, random_state=1)

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)

testing_data = count_vector.transform(x_test)

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
y_pred = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

ip = [input("Enter data")]
dataa = count_vector.transform(ip)
new_pred = naive_bayes.predict(dataa)
print(new_pred)
