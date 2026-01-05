from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def build(self, params):
        """Construire le modèle avec les hyperparamètres"""
        pass

    @abstractmethod
    def suggest_params(self, trial):
        """Retourne l’espace des hyperparamètres pour Optuna"""
        pass

    @abstractmethod
    def fit(self, train_df):
        """Entraîner le modèle sur les données"""
        pass

    @abstractmethod
    def predict(self, test_df):
        """Faire des prédictions"""
        pass
