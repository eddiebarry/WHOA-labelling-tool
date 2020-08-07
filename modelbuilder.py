import sys
sys.path.insert(1, './models/vac_safety')

from model import VacSafetyModel as vac_safety_model

# A class where we can extensibly add models as they are created
class ModelFinder:
    def __init__(self, project_name=None):
        self.name      = project_name

    def getModel(self):
        if self.name == "vac_safety":
            return vac_safety_model()
        else:
            return None
