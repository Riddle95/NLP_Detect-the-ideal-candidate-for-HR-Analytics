import pandas as pd
import numpy as np
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# list of stopwords
stopwords = set(stopwords.words('english'))

# Matching the key words to a path of string
    # Detect the ideal candidate: HR Analytics
        # Comparison:
            # Job Description: ICT Business Analysis_AI Australia
            # Answers of candidate: data sample from MIT's interview question_question 6
        # Questions:
            # Q.1: how you doing?
            # Q.2: tell me about yourself
            # Q.3: Tell me about a time when you demonstrated leadership?
            # Q.4: Tell me about a time when you were working on a team and you were faced with a challenge. How did you solve the problem?
            # Q.5: Tell me about one of your weaknesses and how you plan to overcome it?
            # Q.6: Now why do you think that we should hire you for this position?
        # Note:
            # Sometimes the interviewer asks follow-up question to clarify the answer of interviewee
            # Unsure whether the reactions of interviewer will affect to the performance of interviewee

# Direction:
    # Apply cosine_similarity with tf-idf to define the matches candidates
        # Sample size (n) = 48 candidates
        # Result: csim's score - range: [0,1] ;
            # closer to 1 ~ better
                # Review the last row in csim for observing the result
                    # last row = compare the matching level of the job description to the answer for the question 6 of each candidates

# Further exploration (For improvement):
    # Define the matching level into several components:
        # Identify the score for each components
            # I.e. components = communication
                # Create the bag of certain keywords identifying the level of communication
                # compare the answers with that bag



def script_break(listofstring, sub_string):
    list = []
    for j in train:
        for i in j.split('|'):
            if i.startswith(sub_string):
                list.append(i)
    return list

if __name__ == '__main__':
    df = pd.read_csv('/Volumes/New/interview_transcripts_by_turkers.csv')
    train = df.iloc[0:50, 1]
    test = df.iloc[51:, 1]
    listoftrain = train.tolist()
    listoftest = test.tolist()
    sub_string_1 = "Interviewer"
    sub_string_2 = "Interviewee"
    list_train_interviewer = script_break(listoftrain, sub_string_1)
    list_train_interviewee = script_break(listoftrain, sub_string_2)
    print(list_train_interviewer)
    print(list_train_interviewer)

# dictionary of lists
dict = {'interviewer': list_train_interviewer}

dict_2 = {'interviewee': list_train_interviewee}

df_export = pd.DataFrame(dict)
df_export_2 = pd.DataFrame(dict_2)

# saving the dataframe
df_export.to_csv('Interviewer.csv')
df_export_2.to_csv('Interviewee.csv')

# Sorting data into certain categories, breaking by question type

def pick(list, column, elements):
    new_list = list[list[column] == elements]
    return new_list

def tokenized_sentence(answer):
    words = [word_tokenize(i) for i in answer]
    return words

def remove_stopwords(data):
    output_array = []
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for sentence in data:
        temp_list = []
        for i in sentence.split():
            if i.lower() not in stopwords and i not in punc:
                temp_list.append(i)
        output_array.append(' '.join(temp_list))
    return output_array

def remove_stopwords_list(input):
    l = []
    for i in tokenized_sentence(str(j) for j in input):
        new = remove_stopwords(i)
        l.append(new)
    return l

def clean(input):
    l = []
    for i in input:
        data = list(filter(None, i))
        l.append(data)
    return l

def frequency(input):
    output = Counter(x for sublist in input for x in sublist)
    return output

def combine_dict(input_1,input_2):
    new_dict = [input_1, input_2]
    output = {k:v for x in new_dict for k,v in x.items()}
    return output

def docTermDf_create(docs, vocab):
    termDict = {}
    docsTFMat = np.zeros((len(docs),len(vocab)))
    docsIdfMat = np.zeros((len(vocab),len(docs)))
    docTermDf = pd.DataFrame(docsTFMat, columns=sorted(vocab.keys()))
    docCount = 0
    for doc in docs:
        doc = doc.translate(str.maketrans("","",string.punctuation))
        words = word_tokenize(doc.lower())
        for word in words:
            if (word in vocab.keys()):
                docTermDf[word][docCount] += 1
        docCount+=1
    return docTermDf


if __name__ == '__main__':
    df_read = pd.read_csv('/Volumes/New/Python_practice_new/Python_Practice/Interviewer.csv')
    column = 'Q'
    df_jd = pd.read_csv('/Volumes/New/JD_ICT_BA.csv')
    jd = df_jd.iloc[0]
    Q1 = 1
    Q2 = 2
    Q3 = 3
    Q4 = 4
    Q5 = 5
    Q6 = 6
    list_q1 = pick(df_read, column, Q1).iloc[:,3]
    list_q2 = pick(df_read, column, Q2).iloc[:,3]
    list_q3 = pick(df_read, column, Q3).iloc[:,3]
    list_q4 = pick(df_read, column, Q4).iloc[:,3]
    list_q5 = pick(df_read, column, Q5).iloc[:,3]
    list_q6 = pick(df_read, column, Q6).iloc[:,3]
    list_q1_split = clean(remove_stopwords_list(list_q1))
    list_q2_split = clean(remove_stopwords_list(list_q2))
    list_q3_split = clean(remove_stopwords_list(list_q3))
    list_q4_split = clean(remove_stopwords_list(list_q4))
    list_q5_split = clean(remove_stopwords_list(list_q5))
    list_q6_split = clean(remove_stopwords_list(list_q6))
    jd_clean = clean(remove_stopwords_list(jd))
    bagsofwords_q1 = frequency(list_q1_split)
    bagsofwords_q2 = frequency(list_q2_split)
    bagsofwords_q3 = frequency(list_q3_split)
    bagsofwords_q4 = frequency(list_q4_split)
    bagsofwords_q5 = frequency(list_q5_split)
    bagsofwords_q6 = frequency(list_q6_split)
    bagsofwords_jd = frequency(jd_clean)
    vocab_jdq6 = combine_dict(bagsofwords_jd, bagsofwords_q6)
    q6_new = pick(df_read, column, Q6)
    q6_new_new = q6_new[['Code', 'interviewee']]
    A = q6_new_new.groupby('Code')['interviewee'].apply(list)
    A = A.tolist()
    A1 = [''.join(str(x)) for x in A]
    A2 = A1 + jd.tolist()
    doctermdf_jdq6 = docTermDf_create(A2, vocab_jdq6)
    csim = cosine_similarity(doctermdf_jdq6, doctermdf_jdq6)
    print(csim)

