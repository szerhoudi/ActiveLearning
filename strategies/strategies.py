import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, ProbabilisticModel
from libact.utils import zip
from sklearn.metrics.pairwise import cosine_similarity


class CMBSampling(QueryStrategy):

    def __init__(self, *args, **kwargs):
        super(CMBSampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        self.lmbda = kwargs.pop('lmbda', 0.5)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )

        self.model.train(self.dataset)

    def make_query(self, return_score=False):

        dataset = self.dataset
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        if np.shape(dvalue)[1] > 2:
            dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])

        dist = np.abs(dvalue[:, 0] - dvalue[:, 1])
        arr1, arr2 = [], []
        arr1.append(np.array(dvalue[:, 0]).tolist())
        arr2.append(np.array(dvalue[:, 1]).tolist())

        div = -np.max(cosine_similarity(arr1, arr2), axis=1)
        score = (self.lmbda * dist) + ((1 - self.lmbda) * div)

        ask_id = np.argmin(score)

        if return_score:
            return unlabeled_entry_ids[ask_id], \
                   list(zip(unlabeled_entry_ids, score))
        else:
            return unlabeled_entry_ids[ask_id]


class USampling(QueryStrategy):

    def __init__(self, *args, **kwargs):
        super(USampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )

        self.model.train(self.dataset)
        self.method = kwargs.pop('method', 'mm')

    def make_query(self, return_score=False):

        dataset = self.dataset
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        '''if self.method == 'sm': 
            if np.shape(dvalue)[1] > 2:
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])'''

        if self.method == 'mm':  # max margin
            margin = np.partition(-dvalue, 1, axis=1)
            score = -np.abs(margin[:, 0] - margin[:, 1])

        '''elif self.method == 'entropy':
            score = np.sum(-dvalue * np.log(dvalue), axis=1)'''

        ask_id = np.argmax(score)

        if return_score:
            return unlabeled_entry_ids[ask_id], \
                   list(zip(unlabeled_entry_ids, score))
        else:
            return unlabeled_entry_ids[ask_id]
