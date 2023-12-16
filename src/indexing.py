import heapq
import os

import nltk
import string
from typing import List

import numpy as np

from src.dataset import Documents


class Term:
    """
    In this class, I implement a term object, which store the valuse of a term(str), the list of documents that it is
    occurred in, and its index in them, the cf and df.
    ...
    Attributes:
    ----------
    word: str
        value of a token
    docs: list[dict]
        list of documents that it is occurred in, and its index in them. Each document is represented as a dict with doc_idx as key(index of the document) and
        a list of indexes as value( list of index of the occurrence of the token in this document)
    """

    def __init__(self, word: str):
        self.word: str = word
        self.docs: list[dict] = []
        self.df: int = 0
        self.cf: int = 0

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word


class InvertedIndex:
    """
    In this class, I implement an information retrieval system which can search a query among documents.
    ...
    Attributes:
    -----------
    collection_size: int
        number of documents.
    documents: List
        list of documents in format of Document object.
    posting_list: List[Token]
        list of Term objects. Terms store a string , document's indexes , cf and df
    tf_idf_matrix: List[List] (matrix)
        The tf-idf matrix that described in the slides.


    Methods
    -------
    Methods defined here:
        __init__(self, documents: List, case_sensitive=False):
            Constructor will set initial attributes like case_sensitivity. NOTE that documents should be read with
            read_document function after creating our IR system.
            :parameter
            ---------
            :return
                None

        get_token_index(self, x):
            this function finds index of a word in posting list using binary search algorithm.
            :parameter
                x:str
                    the word you want to find its index
            :return
                int: index of the word in posting_list

        create_posting_list(self):
            calling this function, will create posting list of all occurred words cross all documents.
            :parameter
                None
            :return
                None

        create_tf_idf_matrix(self):
            This function will create a tf-idf matrix. I used the formula in the slides.
            :parameter
                None
            :return
                None
    """

    def __init__(self, documents, collection_size):
        self.collection_size: int = collection_size
        self.documents: list[Documents] = documents
        self.tf_idf_matrix = []
        self.posting_list: List[Term] = []

    def get_term_index(self, term):
        """
        This function find index of a word in posting list using binary search algorithm.
            :parameter
                x:str
                    the word you want to find its index
            :return
                int: index of the word in posting_list
        """
        low = idx = 0
        high = len(self.posting_list) - 1
        while low <= high:
            if high - low < 2:
                if self.posting_list[high].word < term:
                    idx = high + 1
                    break
                elif self.posting_list[high].word == term:
                    idx = high
                    break
                elif self.posting_list[low].word >= term:
                    idx = low
                    break
            idx = (high + low) // 2
            if self.posting_list[idx].word < term:
                low = idx + 1
            elif self.posting_list[idx].word > term:
                high = idx - 1
            else:
                break
        return idx

    def create_posting_list(self):
        """
        calling this function, will create posting list of all occurred words cross all documents. in this function, I
        loop over all documents, then inside this loop, I loop over all the tokens that are in the current document.
        then I check if the length of posting_list is zero, then I add this token as first term. else if the length of
        posting_list is more than 0, I find the correct index of the token in posting_list alphabetically. then I check
        if this token, has been already in posting_list, I just add the current document index in tokens.docs, else, I
        add this token in the posting_list, then add the current document index. I also calculate cf and df during the loops.
            :parameter
                None
            :return
                None
        :return:
        """
        for doc_idx, doc in enumerate(self.documents):
            for token_idx, token in enumerate(doc.text):
                if len(self.posting_list) == 0:
                    self.posting_list.append(Term(token))
                    self.posting_list[0].docs.append({doc_idx: [token_idx]})
                    self.posting_list[0].cf += 1
                    continue

                idx = self.get_term_index(token)

                if idx == len(self.posting_list):
                    self.posting_list.append(Term(token))
                    # self.posting_list[i].post_idx.append(post_idx)
                elif token != self.posting_list[idx].word:
                    self.posting_list.insert(idx, Term(token))

                if len(self.posting_list[idx].docs) == 0:
                    self.posting_list[idx].docs.append({doc_idx: [token_idx]})
                    self.posting_list[idx].df += 1
                elif doc_idx not in self.posting_list[idx].docs[-1].keys():
                    self.posting_list[idx].docs.append({doc_idx: [token_idx]})
                    self.posting_list[idx].df += 1
                else:
                    self.posting_list[idx].docs[-1][doc_idx].append(token_idx)
                self.posting_list[idx].cf += 1

    def idf(self, df_t):
        return np.log(self.collection_size / df_t)

    def create_tf_idf_matrix(self):
        """
        This function will create a tf-idf matrix. I used the formula in the slides. Fisrt I set all values of the matrix to zeros then I loop over all terms in posting list and then loop over all documents in each term, an set the row of t-th term and doc_idx-th column to tf*idf.
        :return:
            None
        """
        self.tf_idf_matrix = np.zeros([len(self.posting_list), len(self.documents)])
        for t in range(len(self.posting_list)):
            for doc in self.posting_list[t].docs:
                doc_idx, indexes = next(iter(doc.items()))
                self.tf_idf_matrix[t, doc_idx] = len(doc[doc_idx]) * np.log(
                    self.collection_size / self.posting_list[t].df)

        for col_idx in range(len(self.documents)):
            v_norm = np.linalg.norm(self.tf_idf_matrix[:, col_idx])
            if v_norm != 0:
                self.tf_idf_matrix[:, col_idx] = self.tf_idf_matrix[:, col_idx] / v_norm
