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
