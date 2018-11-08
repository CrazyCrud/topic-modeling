import gensim
import random
import os
import pickle
import glob
from preprocessing import prepare_text_for_lda


def get_text_data():
    text_data = []
    text_labels = []
    txt_files = glob.glob('corpus/*.txt')

    for document in txt_files:
        file = open(document, 'r')

        tokens = prepare_text_for_lda(file.read())

        if random.random() > .70:
            print(tokens)

        text_data.append(tokens)

        head, tail = os.path.split(document)
        text_labels.append(os.path.splitext(tail)[0])

    return text_data, text_labels


def find_topics(lda_model):
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)
    return lda_model


def get_document_topics(lda_model, corpus):
    return lda_model.get_document_topics(corpus, minimum_probability=0.1)


def create_similarity_matrix(corpus):
    tfidf_transform_tool = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf_transform_tool[corpus]

    index = gensim.similarities.MatrixSimilarity(tfidf_transform_tool[corpus])
    similarities = index[corpus_tfidf]

    """
    Print the similarity of one document to all others
    """
    print(list(enumerate(similarities)))

    """
    Print sorted (document number, similarity score) 2-tuples
    """
    # similarities = sorted(enumerate(similarities[1]), key=lambda item: -item[1])
    # print(similarities)


if __name__ == "__main__":
    text_data, text_labels = get_text_data()

    """
    The dictionary contains unique tokens found in the document set
    """
    dictionary = gensim.corpora.Dictionary(text_data)

    """
    The tokens of the documents are stored in the variable corpus.
    They are stored in tuples where the id of the specific word is the first index
    and the second index represents the word count.
    corpus[10] = [(12, 3), (14, 1), ...] 
    """
    corpus = [dictionary.doc2bow(text) for text in text_data]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    NUM_TOPICS = 3
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    lda_model.save('model{}.gensim'.format(NUM_TOPICS))

    find_topics(lda_model=lda_model)

    all_topics = get_document_topics(lda_model=lda_model, corpus=corpus)
    for index, doc_topics in enumerate(all_topics):
        print('{}'.format(text_labels[index]))
        print('Document topics: {}'.format(doc_topics))
        print('\n')

    create_similarity_matrix(corpus=corpus)

    # visualize_lda(lda=lda_model, corpus=corpus, dictionary=dictionary)
