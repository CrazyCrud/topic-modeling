import gensim
import os
import pickle
import glob
from preprocessing import prepare_text_for_lda
from visualize import visualize_lda


def get_text_data():
    """
    Save the text data and the txt. file names in lists
    """
    text_data = []
    text_labels = []
    txt_files = glob.glob('corpus/*.txt')

    for document in txt_files:
        file = open(document, 'r')
        tokens = prepare_text_for_lda(file.read())
        text_data.append(tokens)

        head, tail = os.path.split(document)
        text_labels.append(os.path.splitext(tail)[0])

    return text_data, text_labels


def generate_lda_model(corpus, dictioanry, number_of_topics):
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=number_of_topics, id2word=dictionary, passes=15)
    lda_model.save('model{}.gensim'.format(NUM_TOPICS))
    return lda_model


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


def save_corpus(corpus):
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


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

    """
    Save the corpus so it can be loaded to save some time
    """
    save_corpus(corpus)

    """
    Create the topic model
    """
    NUM_TOPICS = 5
    lda_model = generate_lda_model(corpus=corpus, dictioanry=dictionary, number_of_topics=NUM_TOPICS)

    """
    Find topics in the model
    """
    find_topics(lda_model=lda_model)

    """
    Get the topics of each document    
    """
    all_topics = get_document_topics(lda_model=lda_model, corpus=corpus)
    for index, doc_topics in enumerate(all_topics):
        print('{}'.format(text_labels[index]))
        print('Document topics: {}'.format(doc_topics))
        print('\n')

    """
    Look at the similarity of the documents
    """
    create_similarity_matrix(corpus=corpus)

    """
    Visualize the topic model
    """
    visualize_lda(lda=lda_model, corpus=corpus, dictionary=dictionary)
