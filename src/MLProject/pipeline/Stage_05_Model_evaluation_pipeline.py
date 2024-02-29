from MLProject.config.configuration import ConfigurationManager
from MLProject.components.Model_evaluation import ModelEvaluation
from MLProject import logger

STAGE_NAME = "MODEL EVALUATION STAGE"


class ModelEvaluationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evalution_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()

if __name__=='__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<<")
        modelevaluation = ModelEvaluationTrainingPipeline()
        modelevaluation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx========")

    except Exception as e:
        logger.exception(e)
        raise e