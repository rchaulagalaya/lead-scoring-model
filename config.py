import os
from common import DataSource, ClassifierType

class Config:
    # Environment configurations
    ENV = os.getenv("ENV", "local")  # Default to "Dev" if ENV is not set  
    JSON_FILE_PATHS = {
        "testing": "sample-data/plf_lead_events_test_data.json", # mode this to env settings
        "training": "sample-data/plf_lead_events_raw_prod_original.json",
    }
    DB_ENV={
        "local": "mongodb://localhost:27017/",
        # "testing_db":  "mongodb+srv://mongouser:H8zYgzuQbTb5psyc@cluster0-dk5qf.mongodb.net/",
        "testing_db":  "mongodb+srv://mongouser:mPtojUhz4TadduZ6@cluster0-8zgt8.mongodb.net/test?retryWrites=true&w=majority", # prod
        "training_db": "mongodb+srv://mongouser:mPtojUhz4TadduZ6@cluster0-8zgt8.mongodb.net/test?retryWrites=true&w=majority" # prod
        }
    DATA_SOURCE = os.getenv("DATA_SOURCE", "JSON") # Default to JSON if not set
    DATA_COUNT= os.getenv("DATA_COUNT","200000") # Default 200000
    UNDER_SAMPLE_MAJORITY=os.getenv("UNDER_SAMPLE_MAJORITY", "False").lower() == "true"  # Converts to boolean #os.getenv("UNDER_SAMPLE_MAJORITY", True) # Default False
    CLASSIFIER_TYPE= os.getenv("CLASSIFIER_TYPE",ClassifierType.BINARY) # Default BINARY
    BINARY_CLASSIFIER_MODEL_NAME = os.getenv("BINARY_CLASSIFIER_MODEL_NAME", "")


    @staticmethod
    def get_data_source():
        """Get the data source setting (JSON or DATABASE)."""
        data_source = Config.DATA_SOURCE
        print(f"data_source: {data_source}")
        if data_source in DataSource.__members__:  # This checks if the value exists in the enum's member names
            return DataSource[data_source]  # This converts the string to the corresponding enum member
        else:
            raise ValueError(f"Invalid DATA_SOURCE found in env settings: {Config.DATA_SOURCE}. Expected one of: {', '.join(DataSource.__members__.keys())}")

    @staticmethod
    def get_data_count():
        return  int(Config.DATA_COUNT)    

    @staticmethod
    def get_under_sample_flag():
        return  Config.UNDER_SAMPLE_MAJORITY

    @staticmethod
    def get_binary_classifier_model_name():
        """Get the binary classifier model name from the environment variable."""
        return Config.BINARY_CLASSIFIER_MODEL_NAME

    @staticmethod
    def get_json_file_path(purpose: str) -> str:
        """Get the JSON file path for a given purpose ('testing' or 'training')."""
        if purpose not in Config.JSON_FILE_PATHS:
            raise ValueError(
                f"Invalid purpose '{purpose}' provided. Expected one of: {', '.join(Config.JSON_FILE_PATHS.keys())}"
            )
        return Config.JSON_FILE_PATHS[purpose]

    @staticmethod
    def get_db_server(purpose: str) -> str:
        """Get the Database environment (local, dev, prod, uat, test) from the environment variable."""
        if purpose not in Config.DB_ENV:
            raise ValueError(
                f"Invalid purpose '{purpose}' provided. Expected one of: {', '.join(Config.DB_ENV.keys())}"
            )
        return Config.DB_ENV[purpose]       
    
    @staticmethod
    def get_classifier_type():
        """Get the classifier type based on the environment variable."""
        classifier_type=Config.CLASSIFIER_TYPE
        if classifier_type in ClassifierType.__members__:
            return ClassifierType[classifier_type]
        else:
           raise ValueError(f"Invalid CLASSIFIER_TYPE: {Config.CLASSIFIER_TYPE} found in env settings. Expected one of: {list(ClassifierType.__members__.keys())}")
     
    _logger = None 
    @staticmethod
    def configure_logger():
        import logging
        from logging.handlers import RotatingFileHandler

        logger = logging.getLogger(__name__)  # Module-specific logger
        logger.setLevel(logging.INFO)

        if not logger.handlers:  # Prevent duplicate handlers
            # File handler
            log_file_grows_indifinitely = True
            if log_file_grows_indifinitely:
                file_handler = RotatingFileHandler("classifier_logs_growing.txt", maxBytes=5*1024*1024, backupCount=5)
            else:
                file_handler = logging.FileHandler("classifier_logs.txt")

            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

            # Stream handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)

        return logger

    @staticmethod
    def get_logger():
        if Config._logger is None:
            Config._logger = Config.configure_logger()
        return Config._logger




