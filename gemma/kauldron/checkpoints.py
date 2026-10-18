import abc


class AbstractPartialLoader(abc.ABC):
    @abc.abstractmethod
    def transform(self, state):
        raise NotImplementedError
