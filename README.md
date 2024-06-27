# Information Retrieval: Document Ranking

This project focuses on implementing and evaluating various information retrieval models, including the Okapi BM25, Cosine Similarity, and Language Models. The aim is to compare the performance of these models in terms of precision and recall, and to determine which model performs best under different conditions.

## Description

The project involves preprocessing text data, implementing ranking models, and evaluating their performance using precision-recall metrics. The models are tested with various queries to analyze their effectiveness in retrieving relevant documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- matplotlib
- heapq

You can install the required libraries using the following command:

```bash
pip install numpy matplotlib
```

## Usage

To use the models, you need to preprocess your text data and create an instance of the respective model class. Here's an example of how to use the Okapi BM25 model:

```python
from src.preprocessing import preprocess_documents
from src.indexing import InvertedIndex
from src.ranking import OkapiBM25
from src.utils import read_documents

documents, collection_size = read_documents(path='../dataset/raw/cran.all.1400')
preprocessed_documents = preprocess_documents(documents)

inverted_index = InvertedIndex(documents=preprocessed_documents,
                               collection_size=collection_size)
inverted_index.create_posting_list()
inverted_index.create_tf_idf_matrix()

# Initialize the model with your inverted index
okapi_bm25 = OkapiBM25(inverted_index)

# Compute rankings for a given query
results = okapi_bm25.compute_rankings(query="example query", k_top=10, k1=1.5, k3=1.5, b=0.75, method='long')

# Print the top results
print(results)
```

## Models Implemented

### Okapi BM25

Okapi BM25 is a probabilistic information retrieval model that ranks documents based on their relevance to a query. The model parameters include k1, k3, and b, which control term frequency saturation, term frequency in the query, and document length normalization, respectively.

### Cosine Similarity

Cosine Similarity measures the cosine of the angle between two vectors representing the query and the document. It is used to determine the similarity between the query and the document.

### Language Model

The Language Model approach ranks documents based on the probability of generating the query given the document. The model uses a smoothing parameter lambda to balance the probability estimates.

## Evaluation

The models are evaluated using precision-recall metrics. The evaluation involves running a set of queries against the models and calculating precision and recall at various points. An example of evaluating the Okapi BM25 model is shown below:

```python
from src.utils import read_queries
query_list, relevant_list = read_queries(X_path='../dataset/raw/cran.qry', y_path='../dataset/raw/cranqrel', n=225)

interpolated_precision_list = []
sample_queries = query_list[0:5]
sample_relevants = relevant_list[0:5]

for i,query in enumerate(sample_queries):
    # curr_results = mapping_model_func[model_name](query=query, k_top=k_top, **model_param)
    sample_results = okapi_bm25.compute_rankings(query, k_top=10, k1=1.5, k3=1.5, b=0.75, method='long')
    curr_interpolated_precision = eval_11_point_interpolation(y_orig=sample_relevants[i], predicted=[doc[1] for doc in sample_results])
    interpolated_precision_list.append(curr_interpolated_precision)
recall = [i*0.1 for i in range(11)]
mean_average_precision = np.array(interpolated_precision_list).sum(axis=0)/len(sample_queries)

plt.plot(recall, mean_average_precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()
```

## Results

In the final evaluation, the Okapi BM25 model demonstrated the highest precision, followed by the Cosine Similarity model and the Language Model. The Okapi BM25 model is recommended for tasks requiring high precision in information retrieval.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.