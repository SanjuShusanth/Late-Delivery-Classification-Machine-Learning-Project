from MLProject.config.configuration import ConfigurationManager
from MLProject.components.data_transformation import DataTransformation
from MLProject import logger

STAGE_NAME = "DATA TRANSFORMATION STAGE"


class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.get_data_transformation_object()
        data_transformation.initiate_data_transformation() 

if __name__=='__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<<")
        datatransformation = DataTransformationTrainingPipeline()
        datatransformation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<<\n\nx========")

    except Exception as e:
        logger.exception(e)
        raise e