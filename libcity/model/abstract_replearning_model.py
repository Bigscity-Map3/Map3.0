class AbstractRepLearningModel:

    def __init__(self, config, data_feature):
        self.config=config
        self.data_feature = data_feature

    def run(self, train_dataloader=None,eval_dataloader=None):
        """
        Args:
            data : input of tradition model

        Returns:
            output of tradition model
        """

    def save_model(cache_name):
        """
        Args:
            cache_name : path to save parameters
        """
    
    def load_model(cache_name):
        """
        Args:
            cache_name : path to load parameters
        """
    
    def get_representation():
        """
        Returns:
            node embedding or model parameters
        """
    
