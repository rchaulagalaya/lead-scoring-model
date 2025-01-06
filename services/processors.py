from enum import Enum
from datetime import datetime, timedelta, timezone
import logging
from multiprocessing import process
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta
from services import mongo_service as mongoDbService
from  config import  Config
import json
config=Config()

class data_provider:
    def __init__(self, logger):
        self.logger = logger
        self.db = mongoDbService.dbConnection(logger,config)


    def format_date(self,date_str):
        """ Formats to string date. removes timezone.
               Parse the date and format it to the desired string format
        """
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return "NO DATE"

    def from_json(self,data_count,file_path):
        try:
            json_file_path = file_path
            self.logger.info("Fetching data from Json. File path %s",json_file_path)

            # Get the start of the current month
            now = datetime.now()
            start_of_month = datetime(now.year, now.month, 1)

            # Load JSON data from the file
            with open(json_file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                self.logger.info("Loaded Json data count: %d", len(json_data))

                # Filter data
                filtered_data = []
                for doc in json_data:
                    try:
                        # Parse and compare createdAt dates
                        created_at = doc.get("createdAt", {}).get("$date", "NA")
                        if created_at == "NA":
                            continue  # Skip invalid dates
                        created_at_date = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        ).replace(tzinfo=None)

                        lead_category = doc.get("leadCategory", "NA")
                        lead_source = doc.get("leadSource", "NA")
                        lead_segment = doc.get("leadSegment", "NA")
                        enquiry_id = doc.get("enquiryId", "NA")
                        ff_sent = doc.get("lcr", {}).get("ffSent", {}).get("$date", "NA")
                        # Process ff_sent only if it is not NA
                        if ff_sent != "NA":
                            ff_sent_date = datetime.fromisoformat(ff_sent.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")

                        if (
                            enquiry_id is not None
                            and lead_category in ["LDAS", "INCOMPLETE"]
                            and lead_source not in [None, ""]
                            and lead_segment not in [None, ""]
                            and created_at_date < start_of_month
                            and "lcr" in doc
                        ):
                            filtered_data.append(doc)
                    except Exception as inner_error:
                        self.logger.warning("Error processing document: %s. Skipping it.", inner_error)

                self.logger.info("Filtered Json data count: %d", len(filtered_data))

                 # Process and standardize the fetched data
                processed_data = []
                for f_data in filtered_data:
                    fff_sent_date = f_data.get("lcr", {}).get("ffSent", {}).get("$date", "NA")
                    if isinstance(fff_sent_date, datetime):
                        fff_sent_date = datetime.fromisoformat(fff_sent_date.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")

                    processed_data.append({
                        "_id": f_data.get("_id"),
                        "loanAmount": f_data.get("loanAmount", "NA"),
                        "leadCategory": f_data.get("leadCategory", "NA"),
                        "leadSource": f_data.get("leadSource", "NA"),
                        "leadSegment": f_data.get("leadSegment", "NA"),
                        "lcr.ffSent": fff_sent_date
                    })

                self.logger.info("Processed Json data count: %s", len(processed_data))
                return processed_data
            
        except Exception as e:
            self.logger.error("An error occurred while reading the JSON data: %s from path %d", e,json_file_path)

    def from_db(self, data_count,purpose):
        """Get the Query Data from database.
            Args : total number of data
            Returns : Collection of queried db data."""
        try:
            self.logger.info("Fetching data from Database.")
            self.logger.info(f"DB Purpose. {purpose}")
            processed_data = []

            # Get the start of the current month
            #choose training data month or 2 ago?
            now = datetime.now()
            one_months_ago = now - relativedelta(months=1)
            two_months_ago = now - relativedelta(months=2)
            start_of_two_months_ago = datetime(two_months_ago.year, two_months_ago.month, 1)
            start_of_one_months_ago = datetime(one_months_ago.year, one_months_ago.month, 1) 

            # Connect to MongoDB
            client = self.db.get_mongo_connection(purpose)
            if client is None:
                self.logger.error("MongoDB connection failed.")
                return processed_data

            database = client["policymatrix"]
            collection = database["plf_lead_events"]

             # Define the filter, projection, and sorting
            filter_query = {
                "enquiryId": {"$exists": True}  #  enquiryId is present
                ,"leadCategory": {"$in": ["LDAS", "INCOMPLETE"]} #  leadCategory
                ,"leadSource":{"$exists": True, "$ne": None}
                ,"leadCategory":{"$exists": True, "$ne": None}
                ,"leadSegment":{"$exists": True, "$ne": None}        
            }

            # Modify the filter based on the purpose
            if purpose == "training_db":
                filter_query["createdAt"] = {"$lt": start_of_two_months_ago}  # Get data before 2 months ago
            elif purpose == "testing_db":
                filter_query["createdAt"] = {"$gt": start_of_two_months_ago}  # Get data after 2 months ago
                # filter_query["lcr"] = {"$exists": False}  # Ensure lcr exists

            self.logger.info(f"filter_query: {filter_query}")
            projection = {
                "_id": 1,
                "loanAmount": 1,
                "leadCategory": 1,
                "leadSource": 1,
                "leadSegment": 1,
                "lcr": 1  # Nested field
            }
            sort_order = [("createdAt", -1)]
            cursor = collection.find(filter_query, projection).sort(sort_order).limit(data_count)
            if cursor is None:
                self.logger.error("No data retrieved from MongoDB.")
                return processed_data

            # Process and standardize the fetched data
            for doc in cursor:
                ff_sent = doc.get("lcr", {}).get("ffSent", "NA")
                if isinstance(ff_sent, datetime):
                    ff_sent = ff_sent.strftime("%Y-%m-%d %H:%M:%S")  # Convert to string

                processed_data.append({
                    "_id": doc.get("_id"),
                    "loanAmount": doc.get("loanAmount", "NA"),
                    "leadCategory": doc.get("leadCategory", "NA"),
                    "leadSource": doc.get("leadSource", "NA"),
                    "leadSegment": doc.get("leadSegment", "NA"),
                    "lcr.ffSent": ff_sent
                })

            self.logger.info("Retrieved data from Database. Total Count %s", len(processed_data))
            self.db.close_mongo_connection(connection=client)
            return processed_data 
        except Exception as ex:
            self.logger.error("Error while reading data from database: %s", str(ex))
            return None


class data_processor:
    def __init__(self,logger):
        self.logger = logger

    def process_data(self,records):
        """
        Cleans and formats raw data into a structured format.

        Args:
            records (list[dict]): List of raw data records.

        Returns:
            list[dict]: List of processed data dictionaries.
        """
        self.logger.info("Started Processing data. NA None. Data Count %s",len(records))
        processed = []
        for record in records:
            processed.append({
                "ID": record.get("_id", 0),
                "Loan Amount": record.get("loanAmount", "NA") or "NA",
                "Lead Source": record.get("leadSource", "NA") or "NA",
                "Lead Category": record.get("leadCategory", "NA") or "NA",
                "Lead Segment": record.get("leadSegment", "NA") or "NA",
                "FF Sent": record.get("lcr.ffSent", "NA") or "NA",
            })
        self.logger.info("Completed Processing data. NA None. Processed Data Count %s",len(records))
        return processed  

    def encode_categorical_variables(self,df, categorical_columns):
        """
        Encodes categorical variables in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing data to encode.
            categorical_columns (list[str]): List of column names to encode.

        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables.
        """
        self.logger.info("Started Encoding data. Encoding Data Count %s",len(df))
        self.logger.info("Encoding categorical variables/ features. Feature Variable Count %s",len(categorical_columns))
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        self.logger.info("Completed Encoding data. Count %s",len(df))
        return df

    def prepare_features_and_labels(self,df, predictors, target_column='FF Sent'):
        """
        Prepares feature matrix and label vector.

        Args:
            df (pd.DataFrame): Processed DataFrame.
            target_column (str): Name of the target column.
            predictors (list[str]): List of predictor column names.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector.
        """
        self.logger.info("Started Feature Engineering. Preparing features & labels from encoded data count %s",len(df))

        if predictors is None:
            predictors = ['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']

        X = df[predictors]
        y = (df[target_column] != "NA").astype(int)
        self.logger.info("Generated Features (X) %s Labels(y) %d",len(X),len(y))
        return X, y

    def preprocess_data(self,records, categorical_columns, target_column='FF Sent'):
        """
        Full preprocessing pipeline for data.[Clean>DataFrame>Encode>FeatureEngineering]

        Args:
            records (list[dict]): Raw data records.
            categorical_columns (list[str]): List of categorical column names.(features)
            target_column (str): Name of the target column.(labels)

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the label vector.
        """
        # Step 1: Clean and format data
        processed_records = self.process_data(records)

        # Step 2: Convert to DataFrame
        df = pd.DataFrame(processed_records)

        # Step 3: Encode categorical variables
        df = self.encode_categorical_variables(df, categorical_columns)

        # Step 4: Prepare features and labels
        X, y = self.prepare_features_and_labels(df,categorical_columns,target_column)

        return X, y

