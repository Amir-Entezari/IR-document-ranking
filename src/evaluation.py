import numpy as np

from src.ranking import CosineScore, OkapiBM25, LanguageModel


def eval_11_point_interpolation(y_orig, predicted):
    """
    This function calculate the 11-point interpolated average precision. it gets a list of actual relevant
    documents and a list of predicted documents that model returned. Then compute the 11-point interpolated
    average precision.
    :parameter
        y_orig: list
            list of actual relevant documents to a query
        predicted: list
            list of predicted documents that model returned
    :return
        11-point interpolated average precision
    """
    precision = []
    recall = []
    relevant = 0

    for pred in predicted:
        if pred in y_orig:
            relevant += 1
    if relevant == 0:
        relevant = 1
    seen_relevant = 0
    for i, pred in enumerate(predicted):
        if pred in y_orig:
            seen_relevant += 1
        precision.append(seen_relevant / (i + 1))
        recall.append(seen_relevant / relevant)

    interpolated_precision = np.zeros(11)
    # for i in range(10, -1, -1):
    #     j = 0
    #     while recall[j] < i*0.1:
    #         j+=1
    #     interpolated_precision[0: i] = max(precision[j:])
    for i in range(len(predicted) - 1, -1, -1):
        j = 0
        while j <= recall[i] * 10:
            if interpolated_precision[j] < precision[i]:
                interpolated_precision[j] = precision[i]
                j += 1
            else:
                break
    return interpolated_precision


def evaluation(query_list, relevant_list, model_name, model_param, k_top):
    mapping_model_func = {'cosine': CosineScore,
                          'okapi': OkapiBM25,
                          'language-model': LanguageModel}
    interpolated_precision_list = []
    for i, query in enumerate(query_list):
        curr_results = mapping_model_func[model_name](query=query, k_top=k_top, **model_param)
        curr_interpolated_precision = eval_11_point_interpolation(y_orig=relevant_list[i],
                                                                  predicted=[doc[1] for doc in curr_results])
        interpolated_precision_list.append(curr_interpolated_precision)
    # print(interpolated_precision_list)
    return np.array(interpolated_precision_list).sum(axis=0) / len(query_list)
