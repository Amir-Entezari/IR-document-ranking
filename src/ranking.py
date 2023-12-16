import heapq

import numpy as np

from src.preprocessing import preprocess_text


class BaseRankingModel:
    """
        vectorize_query(self, query):
        This function gets a query and create a correspond vector for this query.
        :parameter
            query: str
                The query that you want to vectorize it.
        :return
            np.array
                The vectorized query

    """

    def __init__(self, inverted_index):
        self.inverted_index = inverted_index

    def vectorize_query(self, query):
        """
            This function gets a query and create a correspond vector for this query. I create a vector with length of
            posting list and set all values to zero. The I loop over all terms in my posting list then if this term has
            been occured in query, I set the t-th entry to tf*idf.
            :parameter
                query: str
                    The query that you want to vectorize it.
            :return
                np.array
                    The vectorized query

        """
        preprocessed_query = preprocess_text(query)
        vector = np.zeros(len(self.inverted_index.posting_list))
        for t in range(len(self.inverted_index.posting_list)):
            tf = preprocessed_query.count(self.inverted_index.posting_list[t].word)
            idf = np.log(self.inverted_index.collection_size / self.inverted_index.posting_list[t].df)
            vector[t] = tf * idf
        return vector

    def compute_rankings(self, **kwargs):
        raise NotImplementedError("Subclasses must override this method.")


class CosineScore(BaseRankingModel):
    def compute_rankings(self, query, k_top):
        """
        This function perform cosine similarity scoring for the given query and return top k relevant documents. I do
        the same algorithm that descirbed in slides.
            :parameter
                query: str
                    The query that you want to get relevant documents
                k_top:int
                    top k relevant documents
            :return
                top k relevant documents to given query
        """
        scores = [[0, i + 1] for i in range(len(self.inverted_index.documents))]
        vectorized_query = self.vectorize_query(query)
        for col_idx in range(len(self.inverted_index.documents)):
            scores[col_idx][0] = -1 * np.inner(vectorized_query, self.inverted_index.tf_idf_matrix[:, col_idx])

        heapq.heapify(scores)
        results = []
        for _ in range(k_top):
            score, idx = heapq.heappop(scores)
            results.append([-1 * score, idx])
        # scores.sort(key=lambda x: x[1], reverse=True)
        # results = scores[0:5]
        return results


class OkapiBM25(BaseRankingModel):
    def compute_rankings(self, query, k_top, k1, k3, b, method='basic', long_length=10):
        """
        This function perform Okapi BM25 scoring for the given query and tune parameters and return top k relevant
        documents. First I create a list of 2-tuples with length of all documents to score each documents according to
        the query and set their score to zero. Then I do the exact algorithm in the slides and set the variables. And I
        loop over all documents and calculate the formula that provided in slides, then save them in the correspond
        index in RSV_List.
        :parameter
            query: str
                The query that you want to get relevant documents.
            k_top: int
                Number of most relevant documents
            long_length: int
                bound for length of the query that determine if the query is long or not.
        """
        RSV_list = [[0, i + 1] for i in range(len(self.inverted_index.documents))]
        preprocessed_query = preprocess_text(query)
        N = self.inverted_index.collection_size
        L_avg = np.mean([len(document.text) for document in self.inverted_index.documents])
        L_q = len(query)

        for doc_idx, document in enumerate(self.inverted_index.documents):
            L_d = len(document.text)
            for term in preprocessed_query:
                t = self.inverted_index.get_term_index(term)
                if self.inverted_index.posting_list[t].word == term:
                    tf = document.text.count(term)
                    df = self.inverted_index.posting_list[t].df
                    temp = (np.log(N / df) * (k1 + 1) * tf) / \
                           (k1 * ((1 - b) + b * (L_d / L_avg)) + tf)
                    if L_q >= long_length and method == 'long':
                        temp *= ((k3 + 1) * tf) / (k3 + tf)
                    RSV_list[doc_idx][0] -= temp  # The minus is for using Max-heap

        heapq.heapify(RSV_list)
        results = []
        for _ in range(k_top):
            RSV, idx = heapq.heappop(RSV_list)
            results.append([-1 * RSV, idx])
        return results


class LanguageModel(BaseRankingModel):
    def compute_rankings(self, query, k_top, lambd):
        """
        This function perform a language model scoring for the given query and tune parameters and return top k
        relevant documents. First I create a list of 2-tuples with length of all documents to score each documents
        according to the query and set their score to zero. Then I do the exact algorithm in the slides and set the
        variables. And I loop over all documents and calculate the formula that provided in slides, then save them in
        the correspond index in document_probs.
        """
        document_probs = [[0, i + 1] for i in range(len(self.inverted_index.documents))]
        preprocessed_query = preprocess_text(query)
        T = 0
        for term in self.inverted_index.posting_list:
            T += term.cf

        for doc_idx, document in enumerate(self.inverted_index.documents):
            L_d = len(document.text)
            for i, term in enumerate(preprocessed_query):
                t = self.inverted_index.get_term_index(term)
                tf = document.text.count(term)
                P_t_M_d = tf / (L_d or 1)
                P_t_M_c = self.inverted_index.posting_list[t].cf / T
                P_t_d = lambd * (P_t_M_d) + (1 - lambd) * (P_t_M_c)
                if i == 0:
                    document_probs[doc_idx][0] = P_t_d
                else:
                    document_probs[doc_idx][0] *= P_t_d

        document_probs = [[-prob, i] for prob, i in document_probs]
        heapq.heapify(document_probs)
        results = []
        for _ in range(k_top):
            prob, idx = heapq.heappop(document_probs)
            results.append([-1 * prob, idx])
        return results
