from MLProject.config.configuration import ConfigurationManager
from MLProject.components.data_validation import DataValidation
from MLProject import logger

STAGE_NAME = "DATA VALIDATION STAGE"


class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
        data_validation.initiate_data_split()

if __name__=='__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<<")
        datavalidation = DataValidationTrainingPipeline()
        datavalidation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx========")

    except Exception as e:
        logger.exception(e)
        raise e