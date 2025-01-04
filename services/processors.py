from enum import Enum
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from services import mongo_service as mongoDbService
# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Mongo db.
db = mongoDbService.dbConnection()

import json
from datetime import datetime, timezone

class data_provider:
    def format_date(self,date_str):
        """ Formats to string date. removes timezone.
               Parse the date and format it to the desired string format
        """
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return "NO DATE"

    def from_json(self, data_count,file_path):
        try:
            json_file_path = file_path
            logger.info("Fetching data from Json. File path %s",json_file_path)

            # Get the start of the current month
            now = datetime.now()
            start_of_month = datetime(now.year, now.month, 1)

            # Load JSON data from the file
            with open(json_file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                logger.info("Loaded Json data count: %d", len(json_data))

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
                        logger.warning("Error processing document: %s. Skipping it.", inner_error)

                logger.info("Filtered Json data count: %d", len(filtered_data))

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

                logger.info("Processed Json data count: %s", len(processed_data))
                return processed_data
            
        except Exception as e:
            logger.error("An error occurred while reading the JSON data: %s from path %d", e,json_file_path)

    def from_db(self, data_count):
        try:
            logger.info("Fetching data from Database.")

            # Get the start of the current month
            now = datetime.now()
            start_of_month = datetime(now.year, now.month, 1)

            # Connect to MongoDB
            client = db.get_mongo_connection()
            database = client["policymatrix"]
            collection = database["plf_lead_events"]

             # Define the filter, projection, and sorting
            filter_query = {
                "enquiryId": {"$exists": True}  #  enquiryId is present
                ,"leadCategory": {"$in": ["LDAS", "INCOMPLETE"]} #  leadCategory
                ,"leadSource":{"$exists": True, "$ne": None}
                ,"leadCategory":{"$exists": True, "$ne": None}
                ,"leadSegment":{"$exists": True, "$ne": None}
               , "createdAt": {"$lt": start_of_month}  # Exclude data from the current month
               ,"lcr":  {"$exists": True}
                # , "_id": ObjectId("643402b5bf990396f0251d5b")
                # , "_id": {"$in": [ObjectId("643402b5bf990396f0251d5b"), ObjectId("64341c3bbf990396f0273034"),ObjectId("643434f3bf990396f0292d78"),ObjectId("643474ffbf990396f02e4d0b"),ObjectId("64347db7bf990396f02f00b3"),ObjectId("64348114bf990396f02f4901"),ObjectId("64348d2dbf990396f03043fd"),ObjectId("643497b2bf990396f0311e6e"),ObjectId("64349b2ebf990396f0316941"),ObjectId("643434f3bf990396f0292d78"),]}
            }
            projection = {
                "_id": 1,
                "loanAmount": 1,
                "leadCategory": 1,
                "leadSource": 1,
                "leadSegment": 1,
                "lcr": 1  # Nested field
            }
            sort_order = [("createdAt", -1)]

            # Fetch required data with filter, projection, and sorting
            data = list(
                collection.find(
                    filter_query,
                    projection
                )
                .sort(sort_order)
                .limit(data_count)
            )

            # Process and standardize the fetched data
            processed_data = []
            for doc in data:
                 # Check if ffSent is a datetime object , yes then convert to string
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

            logger.info("Retrieved data from Database. TOTAL %s", len(processed_data))
            db.close_mongo_connection(connection=client)
            return processed_data 
        except Exception as ex:
            logger.error("Error while reading data from database: %s", str(ex))
            return None


class data_processor:

    def process_data(self,records):
        """
        Cleans and formats raw data into a structured format.

        Args:
            records (list[dict]): List of raw data records.

        Returns:
            list[dict]: List of processed data dictionaries.
        """
        logger.info("Started Processing data. NA None. Data Count %s",len(records))
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
        logger.info("Completed Processing data. NA None. Processed Data Count %s",len(records))
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
        logger.info("Started Encoding data. Encoding Data Count %s",len(df))
        logger.info("Encoding categorical variables/ features. Feature Variable Count %s",len(categorical_columns))
        label_encoder = LabelEncoder()
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        logger.info("Completed Encoding data. Count %s",len(df))
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
        logger.info("Started Feature Engineering. Preparing features & labels from encoded data count %s",len(df))

        if predictors is None:
            predictors = ['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']

        X = df[predictors]
        y = (df[target_column] != "NA").astype(int)
        logger.info("Generated Features (X) %s Labels(y) %d",len(X),len(y))
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


class DataSource(Enum):
    JSON = "JSON"
    DATABASE = "DATABASE"

class ClassifierType(Enum):
    BINARY = "BINARY"
    NON_BINARY = "NON_BINARY"