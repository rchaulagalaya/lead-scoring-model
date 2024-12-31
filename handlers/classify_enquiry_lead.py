from datetime import datetime, timezone
import sys, os
import logging
from xml.etree.ElementInclude import include
from numpy import column_stack
import pandas as pd
import numpy as np
from services import mongo_service as mongoDbService


# corelation
import seaborn as sns
import matplotlib.pyplot as plt
import json
from dateutil.parser import parse
from bson import ObjectId

# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Mongo db
db = mongoDbService.dbConnection()

from dateutil.parser import parse

from datetime import datetime, timezone

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# Clean / Format / Parse data "" to NA
def process_data(records):
    processed = []
    for record in records:
        # ID
        id = record.get("_id", 0)
        # logger.info("ID : %s", id)

        # Loan Amount
        loan_amount = record.get("loanAmount", "NA") or "NA"
        # logger.info("loan Amount %s", loan_amount)

        # Lead Source
        lead_source = record.get("leadSource", "NA") or "NA"

        # Lead Segment
        lead_segment = record.get("leadSegment", "NA") or "NA"

        # Lead Category
        lead_category = record.get("leadCategory", "NA") or "NA"

        # FF Sent
        ff_sent = record.get("lcr.ffSent", "NA") or "NA"       
      
        # Format result
        processed.append(
            {
                "ID": id,
                "Loan Amount": loan_amount,
                "Lead Source": lead_source,
                "Lead Category": lead_category,
                "Lead Segment": lead_segment,
                "FF Sent": ff_sent,
            }
        )
    return processed

# Multi-Label Transformation Function
def transform_labels(data):
    """Transform FF Sent column into binary labels and store mapping."""
    label_mapping = {"NA": 0, "FF Sent": 1}  # Custom mapping
    data['FF Sent Binary'] = data['FF Sent'].apply(lambda x: label_mapping.get("FF Sent") if x != "NA" else label_mapping.get("NA"))
    return data, {v: k for k, v in label_mapping.items()}  # Return reverse mapping

def getFromDB(data_count):
    try:
        logger.info("----------Fetching data from Database.")

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

        logger.info("Got data from Db . TOTAL %s", len(processed_data)) 
        # Close MongoDB connection
        db.close_mongo_connection(connection=client)
        return processed_data  # Return filtered and limited DataFrame
    except Exception as ex:
        logger.error("Error while reading data from database: %s", str(ex))
        return None

def get_binary_prediction(data_count):
    try:
        logger.info("Started BINARY classification")
        # Step 1: Get data   
        dbData = getFromDB(data_count)  
    
        #Step 2:  Process / clean / format data
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data)

        # Step 3: Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))      


        # Step 4:  Prepare features and labels ( x & y)
        X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']] 
        y =  (processed_df['FF Sent'] != "NA").astype(int) # if ff sent has value then convert to int

     
         # Step 5: Split data into train and test sets [ 80 % training,  20% test]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Class Distribution:")
        print(y_train.value_counts())
        print("Test Class Distribution:")
        print(y_test.value_counts())

        # Step 6: Ensure X_train and y_train are aligned
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
       

        # Step 7: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # training features
        model.fit(X_train, y_train)  

        #  Step 8: Make predictions
        y_pred = model.predict(X_test)

        # Step 9: Transform the Predicted FFSent (0 or 1)
        actual_predicted_values = ['will_reach' if val == 1 else 'will_not_reach' for val in y_pred]
        # print("Actual Predicted Values:")
        print(actual_predicted_values)

        # Step 10: Evaluate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        return accuracy
    except Exception as ex:
        logger.error(ex)
        return str(ex)

def get_binary_predictionWORKS(data_count):
    try:
        logger.info("Started BINARY classification")

        # Step 1: Get data   
        dbData = getFromDB(data_count)  
    
        #Step 2:  Process / clean / format data
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data)

        # Step 3: Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))      


        # Step 4:  Prepare features and labels ( x & y)
        X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']] 
        y =  (processed_df['FF Sent'] != "NA").astype(int) # if ff sent has value then convert to int


        # Step 5: Separate training and test data 
        valid_ff_sent = processed_df[processed_df['FF Sent'] != "NA"]  # Records with valid ffSent
        invalid_ff_sent = processed_df[processed_df['FF Sent'] == "NA"]  # Records with "NA" ffSent

        # Step 6: Split the training data (mix of "NA" and valid ffSent)
        X_train = pd.concat([X.loc[invalid_ff_sent.index], X.loc[valid_ff_sent.index]])  # Combine "NA" and valid
        y_train = pd.concat([pd.Series([0] * len(invalid_ff_sent)), pd.Series([1] * len(valid_ff_sent))])  # Use 0 for NA and 1 for valid ffSent TODO

        print("Training Class Distribution:")
        print(y_train.value_counts())

        # Step 7: Ensure X_train and y_train are aligned
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

         # Step 8: Split the test data (only valid ffSent)
        X_test = valid_ff_sent[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']]
        y_test = (valid_ff_sent['FF Sent'] != "NA").astype(int)

        # Step 8: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # training features
        model.fit(X_train, y_train)  

        #  Step 9: Make predictions
        y_pred = model.predict(X_test)

        # Step 10: Transform the Predicted FFSent (0 or 1)
        actual_predicted_values = ['will_reach' if val == 1 else 'will_not_reach' for val in y_pred]
        # print("Actual Predicted Values:")
        # print(actual_predicted_values)

        # Step 11: Evaluate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        return accuracy
    except Exception as ex:
        logger.error(ex)
        return str(ex)

def get_multi_label_prediction(data_count):
    try:
        logger.info("Started MULTI-LABEL classification")

        # Step 1: Get data   / Process / clean/format data
        dbData = getFromDB(data_count)          
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data)
       
        # Transform labels for multi-label classification
        processed_df, reverse_mapping  = transform_labels(processed_df)


        # Step 3: Encode categorical variables
        label_encoder = LabelEncoder()       
        categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))
          
        # Multi-label encoding (if there are additional labels)
        mlb = MultiLabelBinarizer()
        y = processed_df[['FF Sent Binary']]

        # Step 3: Split Data into Train/Test Sets
        X = processed_df[['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']]
        y = processed_df['FF Sent Binary']  # Binary labels for FF Sent prediction


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)               

        print("training class distribution:")
        print(y_train.value_counts())

        print("test class distribution:")
        print(y_test.value_counts())

        # Step 5: Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Step 6: Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the Model
        # print("Classification Report:")
        # print(classification_report(y_test, y_pred))  
      
        ## this is for original labels 
        y_pred_original = [reverse_mapping[label] for label in y_pred]
        y_test_original = [reverse_mapping[label] for label in y_test] 

        print("Reversed Classification Report:")
        print(classification_report(y_test_original, y_pred_original))


        # Step 7: Transform predictions back to original labels
        # actual_predicted_values = ff_sent_encoder.inverse_transform(y_pred)
        # print("Actual Predicted Values:")
        # print(actual_predicted_values)

        # Step 8: Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")



        return accuracy
    except Exception as ex:
        logger.error(ex)
        return str(ex)

def lambda_handler(event, context):
    try:
        data_count = 2000
        binary_prediction = True 
        output ={}

        if binary_prediction:
            output = get_binary_prediction(data_count)
        else:
            output = get_multi_label_prediction(data_count)

        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Completed Yuhuu !! ",
                "data":  output #"data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))