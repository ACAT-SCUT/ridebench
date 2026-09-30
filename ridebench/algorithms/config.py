from __future__ import annotations

from argparse import ArgumentParser, Namespace

import numpy as np
from torch.nn import Module


class ModelConfig:
    @staticmethod
    def name() -> str:
        """
        :return: the algorithm's name
        """
        raise NotImplementedError()

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """
        :param parser: the parser to add arguments
        """
        pass

    @staticmethod
    def model(input_len: int, pred_len: int, x_dim: int, exo_con_list: list[str],
              exo_dis_dict: dict[str, int], args: Namespace) -> Module | InferenceModel:
        """
        :param input_len: length of time steps to input
        :param pred_len: length of time steps to predict
        :param x_dim: number of endogenous variables
        :param exo_con_list: exogenous continuous variable name list
        :param exo_dis_dict: exogenous discrete variable dictionary, the key is variable name, the value is total items
        :param args: parsed arguments added in ModelConfig.add_arguments

        N: number of samples
        A: length of time steps to input
        B: length of time steps to predict
        F: number of endogenous variables
        C: number of exogenous continuous variables
        D: number of exogenous discrete variables

        :return: a benchmark model with input ((NxAxF), (NxAxC), (NxAxD), (NxBxC), (NxBxD)) and output (NxBxF)
        """
        raise NotImplementedError()

    @staticmethod
    def need_train() -> bool:
        """
        :return: whether this algorithm uses the benchmark training loop
        """
        return True


class InferenceModel:
    is_global_model = False
    fit_uses_validation = False

    def fit_loader(self, train_dataloader, batch_to_numpy, val_dataloader=None) -> InferenceModel:
        return self

    def predict_batch(
        self,
        x_endo: np.ndarray,
        x_exo_con: np.ndarray,
        x_exo_dis: np.ndarray,
        y_exo_con: np.ndarray,
        y_exo_dis: np.ndarray,
    ) -> np.ndarray:
        return self.fit_predict(x_endo, x_exo_con, x_exo_dis, y_exo_con, y_exo_dis)

    def fit_predict(self, x_endo: np.ndarray, x_exo_con: np.ndarray, x_exo_dis: np.ndarray, y_exo_con: np.ndarray, y_exo_dis: np.ndarray) -> np.ndarray:
        x_exo = np.dstack((x_exo_con, x_exo_dis.astype(x_exo_con.dtype)))
        y_exo = np.dstack((y_exo_con, y_exo_dis.astype(y_exo_con.dtype)))
        exo = np.hstack((x_exo, y_exo))
        return self._fit_predict(x_endo, exo)

    def _fit_predict(self, x_endo: np.ndarray, exo: np.ndarray) -> np.ndarray:
        """
        :param x_endo: NxAxF
        :param exo: Nx(A+B)x(C+D)
        :return: NxBxF
        """
        raise NotImplementedError()
