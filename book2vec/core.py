import numpy as np
import json
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Book2VecAnalysis:
    def __init__(self, file_obj=None):
        self.loaded = False
        self.embedding_matrix = None
        self.vocab_size = None
        self.index_to_review = None
        self.index_to_metadata = None

        if file_obj:
            self.load(file_obj)

    def load(self, file_obj):
        json_data = json.load(file_obj)
        self.embedding_matrix = json_data["embedding_matrix"]
        self.vocab_size = json_data["vocab_size"]
        self.index_to_review = {
            int(k): v for k, v in json_data["index_to_review"].items()
        }
        self.index_to_metadata = {
            int(k): v for k, v in json_data["index_to_metadata"].items()
        }
        self.loaded = True

    def get_suggestions(self, idxs: Union[List[int], int]) -> pd.DataFrame:
        """
        Takes a list of book IDs and returns a list of suggestions.
        This is calculated by taking the mean of the closest 100 books to the
        given book IDs.

        :param idxs:
        :return:
        """
        if isinstance(idxs, int):
            idxs = [idxs]

        suggestions = pd.concat(
            {idx: self._get_nearest(idx, 250) for idx in idxs}, axis=1, join="outer"
        )
        suggestions["cumulative"] = suggestions.mean(axis=1)
        suggestions.sort_values(by="cumulative", ascending=False, inplace=True)
        return suggestions[~suggestions.index.isin(idxs)]

    def _get_nearest(self, idx: int, limit: int = None) -> pd.Series:
        """
        Calculates the nearest neighbours of the given book via consine similarity

        :param idx: the book ID
        :param limit: if provided will limit to this number of nearest neighbours
        :return: a DataFrame containing the similarity values with book ID as index
        """
        x = np.array(self.embedding_matrix[idx]).reshape((1, -1))
        y = np.array(self.embedding_matrix)
        similarity_values = cosine_similarity(x, y).T
        similarity_values = pd.Series(similarity_values.flatten())
        if limit:
            similarity_values = similarity_values.sort_values(ascending=False)
            return similarity_values[:limit]

        return similarity_values
