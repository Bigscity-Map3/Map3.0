from libcity.utils import get_evaluator, ensure_dir
from libcity.executor.abstract_executor import AbstractExecutor


class TwoStepExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self.evaluator = get_evaluator(config,data_feature)
        self.config = config
        self.model = model
        self.exp_id = config.get('exp_id', None)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)

    def evaluate(self, test_dataloader=None):
        """
        use model to test data
        """
        self.evaluator.evaluate()
        test_result = self.evaluator.save_result(self.evaluate_res_dir)
        return test_result

    def train(self, train_dataloader=None, eval_dataloader=None):
        """
        use data to train model with config
        """
        return self.model.run()


    def load_model(self, cache_name):
        pass

    def save_model(self, cache_name):
        pass
