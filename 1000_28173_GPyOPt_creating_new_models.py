# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.models.base import BOModel
import numpy as np

class NewModel(BOModel):
   
    """
    General template to create a new GPyOPt surrogate model

    :param normalize Y: wheter the outputs are normalized (default, false)

    """

    # SET THIS LINE TO True of False DEPENDING IN THE ANALYTICAL GRADIENTS OF THE PREDICTIONS ARE AVAILABLE OR NOT
    analytical_gradient_prediction = False

    def __init__(self, normalize_Y=True, **kwargs ):

        # ---
        # ADD TO self... THE REST OF THE PARAMETERS OF YOUR MODEL
        # ---
        
        self.normalize_Y = normalize_Y
        self.model = None

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """
        self.X = X
        self.Y = Y
        
        # ---
        # ADD TO self.model THE MODEL CREATED USING X AND Y.
        # ---
        
        
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        self.X = X_all
        self.Y = Y_all
        
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        
        if self.model is None: 
            self._create_model(X_all, Y_all)
        else: 
            # ---
            # AUGMENT THE MODEL HERE AND REUPDATE THE HIPER-PARAMETERS
            # ---
            pass
                 
    def predict(self, X):
        """
        Preditions with the model. Returns posterior means m and standard deviations s at X. 
        """

        # ---
        # IMPLEMENT THE MODEL PREDICTIONS HERE (outputs are numpy arrays with a point per row)
        # ---
        
        return m, s
    
    def get_fmin(self):
        return self.model.predict(self.X).min()

