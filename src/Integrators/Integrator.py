import abc
import numpy as np

class Integrator():
    __metaclass__ = abc.ABCMeta

    """
    
    Integrator can have multiple passes.
    
    """

    @abc.abstractmethod
    def isMultiStage(self) -> bool:
        """
        Designates if the integrator is a multi-stage or single (evaluation) stage integrator. 
        
        A multi-stage integrator operates in the following cycle:
        1. Evaluation
        2. Predict
        3. Evaluation
        4. Correct

        A single-stage integrator operates in the following cycle:
        1. Predict
        2. Evaluation
        3. Correct
        """
        pass

    @abc.abstractmethod
    def predict(self, dt: float, pA: np.array, damping: float):
        pass

    @abc.abstractmethod
    def correct(self, dt: float, pA: np.array, damping: float):
        pass