from deeplearning.abstract_network import AbstractNetwork

class FeedForwardNetwork(AbstractNetwork):

    """
    Feed Forward Network Class
    """

    def __init__(self,
                 model='sequential'):

        super().__init__(model=model)

    ##################
    # Public Methods #
    ##################

    def train(self, dataset=None):

        pass
    
    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):

        pass

    def _setLoss(self):

        pass

    def _setOptimizer(self):

        pass

