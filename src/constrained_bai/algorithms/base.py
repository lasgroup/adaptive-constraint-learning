from abc import abstractmethod


class CLBAlgorithm:
    @abstractmethod
    def run(self, bandit):
        pass
