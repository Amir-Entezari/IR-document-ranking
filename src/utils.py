import os

from src.dataset import Documents


def read_documents(path=os.getcwd()):
    documents = []
    with open(path) as f:
        articles = f.read().split('\n.I')
        for i, article in enumerate(articles):
            article = article.split('\n.T\n')[1]
            T, _, article = article.partition('\n.A\n')
            A, _, article = article.partition('\n.B\n')
            B, _, W = article.partition('\n.W\n')
            curr_doc = Documents(index=i + 1, title=T, author=A, detail=B, text=W)
            documents.append(curr_doc)
        collection_size = len(articles)
    return documents, collection_size


def read_queries(X_path, y_path, n):
    with open(X_path, 'r') as query_file:
        queries = query_file.read().split('\n.I')
        query_list = []
        for i, query in enumerate(queries):
            query = query.split('\n')
            query_list.append(' '.join(query[2:]))
    with open(y_path, 'r') as result_file:
        lines = result_file.readlines()
        results = [[] for i in range(n)]
        for line in lines:
            query_idx, document, relevancy = line.split()
            if int(relevancy) > 0:
                results[int(query_idx) - 1].append(int(document))
    return query_list, results
