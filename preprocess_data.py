from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(data, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Version simple de la fonction de prétraitement — même principe que l'originale.

    - Convertit `data` en DataFrame si nécessaire.
    - Vérifie que le DataFrame n'est pas vide.
    - Retourne (train, test) après un train_test_split.

    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Jeu de données complet (features + cible si présent).
    test_size : float, optional
        Proportion du jeu de test (défaut 0.2).
    random_state : int | None, optional
        Graine pour reproductibilité.

    Returns
    -------
    train, test : pandas.DataFrame, pandas.DataFrame
        Sous-ensembles train et test.
    """

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            raise TypeError("`data` doit être un pandas.DataFrame ou convertible en DataFrame") from e

    if data.shape[0] == 0:
        raise ValueError("Le DataFrame `data` est vide")

    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    return train, test