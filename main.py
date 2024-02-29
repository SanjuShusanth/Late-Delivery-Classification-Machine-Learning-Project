from MLProject import logger
from MLProject.pipeline.Stage_01_data_ingestion_pipeline import DataIngestionTrainingPipeline
from MLProject.pipeline.Stage_02_data_validation_pipeline import DataValidationTrainingPipeline
from MLProject.pipeline.Stage_03_data_transformation_pipeline import DataTransformationTrainingPipeline
from MLProject.pipeline.Stage_04_Model_trainer_pipeline import ModelTrainerTrainingPipeline
from MLProject.pipeline.Stage_05_Model_evaluation_pipeline import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<<")
    data_transformation = DataTransformationTrainingPipeline() 
    data_transformation.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model trainer training Stage"

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model evaluation Stage"

try:
    logger.info(f'>>>>>>> Stage {STAGE_NAME} started <<<<<<<<')
    model_evalution = ModelEvaluationTrainingPipeline()
    model_evalution.main()
    logger.info(f'>>>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx==========x')
except Exception as e:
    logger.exception(e)
    raise e