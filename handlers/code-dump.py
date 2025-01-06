# from curses import typeahead
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




def process_opportunity_status(histories):
    # Step 1: Get the opportunityStatusHistory field
    opportunity_status_history = histories
    # logger.info("opportunity_status_histories %s ", len(opportunity_status_history))

    # Log the number of entries
    # logger.info("Processing %d entries in opportunityStatusHistory", len(opportunity_status_history))

    most_recent_entry = None  # Default value if no valid data exists

    if opportunity_status_history:
        # Step 2: Filter valid entries with `updateDate`
        valid_entries = [item for item in opportunity_status_history if "updateDate" in item]

        # logger.info("Found %d valid entries with updateDate", len(valid_entries))
        # logger.info(valid_entries)

        if valid_entries:
            # Step 3: Parse and normalize all `updateDate` fields
            for entry in valid_entries:
                # logger.info("VALID history %s", entry)
                if isinstance(entry["updateDate"], str):
                    # Parse string to offset-aware datetime
                    entry["updateDate"] = parse(entry["updateDate"])
                elif isinstance(entry["updateDate"], datetime) and entry["updateDate"].tzinfo is None:
                    # Convert naive datetime to offset-aware (UTC)
                    entry["updateDate"] = entry["updateDate"].replace(tzinfo=timezone.utc)

               # Replace empty strings in 'oldStatus' with 'NA'
            for entry in valid_entries:
                if entry.get("oldStatus", "") == "":
                    entry["oldStatus"] = "NA"

            # Step 4: Sort entries by updateDate (descending)
            sorted_entries = sorted(
                valid_entries,
                key=lambda x: x["updateDate"],
                reverse=True
            )
            sorted_df = pd.DataFrame(sorted_entries)

            # Replace empty strings in 'oldStatus' with 'NA' after sorting
            # sorted_df['oldStatus'] = sorted_df['oldStatus'].replace('', 'NA')
            # print(sorted_df.to_string(index=False))
            # logger.info("SORTED StatusValid Entries %s ", sorted_entries)

            most_recent_entry = sorted_entries[0]  # Select the most recent one
        else:
            logger.info("No valid entries with updateDate found.")
    else:
        logger.info("No opportunity status history available.")

    # Step 5: Extract the most recent oldStatus value
    recent_old_status = (
        most_recent_entry["oldStatus"]
        if isinstance(most_recent_entry, dict) and "oldStatus" in most_recent_entry
        else "NA"
    )
    return recent_old_status

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
    # logger.info(processed)
    return processed

# Multi-Label Transformation Function
def transform_labels0(data):
    """Transform FF Sent column into binary labels."""
    data['FF Sent Binary'] = data['FF Sent'].apply(lambda x: 1 if x != "NA" else 0)
    return data

def transform_labels(data):
    """Transform FF Sent column into binary labels and store mapping."""
    label_mapping = {"NA": 0, "FF Sent": 1}  # Custom mapping
    data['FF Sent Binary'] = data['FF Sent'].apply(lambda x: label_mapping.get("FF Sent") if x != "NA" else label_mapping.get("NA"))
    return data, {v: k for k, v in label_mapping.items()}  # Return reverse mapping

# RETURN 20 data records
def getFromDB(data_count):
    try:
        logger.info("----------Fetching data from Database.")

        # Get the start of the current month
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)

        # Connect to MongoDB
        client = db.get_mongo_connection()  # Mongo server
        database = client["policymatrix"]  # MongoDB database
        collection = database["plf_lead_events"]  # MongoDB collection

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
        sort_order = [("createdAt", -1)]  # Sort by createdAt in descending order
        # sort_order = [("lcr.ffSent", -1)]

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
             # Check if ffSent is a datetime object and convert it to string
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

        # logger.info("Got data from Db . TOTAL %s", len(data))
        logger.info("Got data from Db . TOTAL %s", len(processed_data))
        # Close MongoDB connection
        db.close_mongo_connection(connection=client)
        return processed_data  # Return filtered and limited DataFrame
    except Exception as ex:
        logger.error("Error while reading data from database: %s", str(ex))
        return None

def lambda_handler0(event, context):
    try:

        print("BINARY PREDICTION")
        get_binary_prediction(200)
        return

        logger.info("Started enquiry lead classification")

        # Step 1: Get data    // 50 records
        dbData = getFromDB(200)

        #Step 2:  Process / clean/format data
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data) # logging
        print(processed_df.to_string(index=False))  # logging this is our main data set

        # Step 3: Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))

        # Step 4:  Prepare features and labels ( x & y)
        X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']] # independent variables
        # y =  label_encoder.fit_transform(processed_df['FFSent'].astype(str))
        # Encode the 'FFSent' column and ensure we don't overwrite the existing data
        ff_sent_encoder = LabelEncoder()
        processed_df['FFSent_encoded'] = ff_sent_encoder.fit_transform(processed_df['FFSent'].astype(str))



        # Step 5 : Separate training and test data
        # Training data : include that has ffSent value and ffSent = NA
        # Test data  : include that has FFSent = value
        valid_ff_sent = processed_df[processed_df['FFSent'] != "NA"]  # Records with valid ffSent
        invalid_ff_sent = processed_df[processed_df['FFSent'] == "NA"]  # Records with "NA" ffSent


         # Step 6: Split the training data (mix of "NA" and valid ffSent)
        X_train = pd.concat([X.loc[invalid_ff_sent.index], X.loc[valid_ff_sent.index]])  # Combine "NA" and valid
        # y_train = pd.concat([pd.Series(["NA"] * len(invalid_ff_sent)), valid_ff_sent['FFSent']])
        y_train = pd.concat([pd.Series([ff_sent_encoder.transform(["NA"])[0]] * len(invalid_ff_sent)), valid_ff_sent['FFSent_encoded']])


        # Ensure X_train and y_train are aligned
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

         # Step 8: Split the test data (only valid ffSent)
        X_test = valid_ff_sent[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']]
        # y_test = valid_ff_sent['FFSent']
        y_test = valid_ff_sent['FFSent_encoded']

        # Step 8: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42) # training features
        model.fit(X_train, y_train)

        #  Step 9: Make predictions
        y_pred = model.predict(X_test)


         # Step 10:  Transform the Predicted FFSent
         # Inverse transform the predicted labels to get the actual predicted values
        actual_predicted_values = label_encoder.inverse_transform(y_pred)
        print("Actual Predicted Values:")
        print(actual_predicted_values)

        # Step 11: Evaluate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")



        return {
        "statusCode": 200,
        "headers": {
            "access-control-allow-origin": "*",
        },
        "body": json.dumps(
            {
                "message": "Successfully Completed Yuhuu !! ",
                "data": "data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))

def lambda_handler(event, context):
    try:
        data_count = 1000
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
                "data": "data"
            }
        ),
    }
    except Exception as ex:
        logger.error(ex)
        return dict(statusCode=500, body=str(ex))

def get_binary_prediction(data_count):
    try:
        logger.info("Started BINARY classification")

        # Step 1: Get data    // 50 records
        dbData = getFromDB(data_count)

        #Step 2:  Process / clean/format data
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data) # logging

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

def get_multi_label_prediction0(data_count):
    try:
        logger.info("Started MULTI-LABEL classification")

        # Step 1: Get data   / Process / clean/format data
        dbData = getFromDB(data_count)
        # dbData = read_from_json(data_count)
        # print(dbData)
        # return
        processed_data = process_data(dbData)
        processed_df = pd.DataFrame(processed_data)
        # print(processed_df.to_string(index=False))
        # return



        # Step 3: Encode categorical variables
        label_encoder = LabelEncoder()
        encoders = {}
        categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
        for col in categorical_columns:
            processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))
            # encoders[col] = label_encoder

        # print(processed_df.to_string(index=False))
        # return


        # Step 3: Encode FF Sent
        ff_sent_encoder = LabelEncoder()
        # processed_df['FFSent_encoded'] = ff_sent_encoder.fit_transform(processed_df['FFSent'].astype(str))
        processed_df['FF Sent'] = ff_sent_encoder.fit_transform(processed_df['FF Sent'].astype(str))

        """ Processed df
                         ID                                     Loan Amount  Lead Source  Lead Category  Lead Segment  FF Sent
            674b5a1a43373cebf3494b31          141                3                       5                      11                   878
            674b1e1f43373cebf3442e2a          171                2                       5                      10                   885
            674ab4b243373cebf33b1a3e           59                2                       5                      12                    879
        """

        # print(processed_df.to_string(index=False))
        # return

        # Step 4: Split data into training and test sets
        # Ensure both FFSent = value and FFSent = "NA" are included
        train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42) # , stratify=processed_df['FFSent'])

        # print(train_df.to_string(index=False))
        # print(test_df.to_string(index=False))
        # return

        X_train = train_df[['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']]
        y_train = train_df['FF Sent']

        X_test = test_df[['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']]
        y_test = test_df['FF Sent']

        print("training class distribution:")
        print(y_train.value_counts())

        print("test class distribution:")
        print(y_test.value_counts())

        # Step 5: Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Step 6: Make predictions
        y_pred = model.predict(X_test)

        # Step 7: Transform predictions back to original labels
        actual_predicted_values = ff_sent_encoder.inverse_transform(y_pred)
        print("Actual Predicted Values:")
        print(actual_predicted_values)

        # Step 8: Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        """  # Step 4:  Prepare features and labels ( x & y)
        X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']]

        # Step 5: Separate training and test data
        valid_ff_sent = processed_df[processed_df['FFSent'] != "NA"]   # 16 valid
        invalid_ff_sent = processed_df[processed_df['FFSent'] == "NA"] #184 invalid

         # Step 6: Encode the 'FFSent' values, but only fit on the training data
        ff_sent_encoder = LabelEncoder()
        ff_sent_encoder.fit(valid_ff_sent['FFSent'])

        # Transform the 'FFSent' column into encoded values for both training and test sets
        processed_df['FFSent_encoded'] = ff_sent_encoder.transform(processed_df['FFSent'].astype(str))


        # Step 6: Split the training data (mix of "NA" and valid ffSent)
        X_train = pd.concat([X.loc[invalid_ff_sent.index], X.loc[valid_ff_sent.index]])  # Combine "NA" and valid
        y_train = pd.concat([pd.Series([ff_sent_encoder.transform(["NA"])[0]] * len(invalid_ff_sent)), valid_ff_sent['FFSent_encoded']])

        print("Training Class Distribution:")
        print(y_train.value_counts())

        # Step 7: Ensure X_train and y_train are aligned
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

         # Step 8: Split the test data (only valid ffSent)
        X_test = valid_ff_sent[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']]
        y_test = valid_ff_sent['FFSent_encoded']

        # Step 8: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42) # , class_weight='balanced') # training features
        model.fit(X_train, y_train)

        #  Step 9: Make predictions
        y_pred = model.predict(X_test)

        # Step 10: Transform the Predicted FFSent (0 or 1)

        actual_predicted_values = ff_sent_encoder.inverse_transform(y_pred)
        print("Actual Predicted Values:")
        print(actual_predicted_values)

        # Step 11: Evaluate Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%") """

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


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


import pandas as pd
from datetime import datetime

def read_from_json(data_count):
    try:
        logger.info("----------Fetching data from JSON file.")

       # Load raw JSON data
        with open('sample-data/plf_lead_events_raw_prod.json', 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            # print(raw_data)


         # Normalize JSON and preprocess `createdAt` and nested fields
        processed_data = []
        for record in raw_data:
            id= record.get('_id', {}).get('$oid')
            # print("ID %s ",id)


            created_at = record.get('createdAt', {}).get('$date')
            if created_at:
                created_at = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                print("created_at %s ",created_at)

            # Handle nested field `lcr.ffSent`
            ff_sent = record.get('lcr', {}).get('ffSent', 'NA')
            if isinstance(ff_sent, dict) and '$date' in ff_sent:
                ff_sent = datetime.strptime(ff_sent['$date'], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(ff_sent, datetime):
                ff_sent = ff_sent.strftime("%Y-%m-%d %H:%M:%S")
            # print("ff_sent %s ",ff_sent)

            # print("loanAmount %s ",record.get('loanAmount'))
            # print("leadCategory %s ",record.get('leadCategory'))
            # print("leadSource %s ",record.get('leadSource'))
            # print("leadSegment %s ",record.get('leadSegment'))
            # print("enquiryId %s ",record.get('enquiryId'))

            processed_data.append({
                '_id': record.get('_id', {}).get('$oid', 'NA'),
                'loanAmount': record.get('loanAmount', 'NA'),
                'leadCategory': record.get('leadCategory', 'NA'),
                'leadSource': record.get('leadSource', 'NA'),
                'leadSegment': record.get('leadSegment', 'NA'),
                'enquiryId': record.get('enquiryId', 'NA'),
                'createdAt': created_at,
                'ffSent': ff_sent
            })


        # Convert processed data to a DataFrame
        df = pd.DataFrame(processed_data) # returns all dataset table with na
        # print("processed_data %s")
        # print(df.to_string(index=False))
        # return

        # Get the start of the current month
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)
        print("start_of_month %s", start_of_month)

        return
        # print(pd.to_datetime(df['createdAt']) < start_of_month)
        # Apply filters equivalent to MongoDB query
        filtered_df = df[
            (df['enquiryId'].notnull()) &  # enquiryId exists
            (df['leadCategory'].isin(['LDAS', 'INCOMPLETE'])) &  # leadCategory is in specified list
            (df['leadSource'].notnull()) &  # leadSource exists and is not None
            (df['leadSegment'].notnull()) &  # leadSegment exists and is not None
            (pd.to_datetime(df['createdAt']) < start_of_month)  # createdAt is before the start of the current month
        ]
        print("HERE::::::::::>>>>")
        print(filtered_df)
        # Select specific columns (projection)
        projection_columns = [
            '_id', 'loanAmount', 'leadCategory',
            'leadSource', 'leadSegment', 'lcr.ffSent'
        ]
        filtered_df = filtered_df[projection_columns]

        # Sort the data by createdAt in descending order
        sorted_df = filtered_df.sort_values(by='createdAt', ascending=False)

        # Limit the number of rows to data_count
        limited_df = sorted_df.head(data_count)

        # # Process and standardize the data
        # processed_data = []
        # for _, row in limited_df.iterrows():
        #     # Extract and format nested field lcr.ffSent
        #     ff_sent = row.get('lcr', {}).get('ffSent', 'NA')
        #     if isinstance(ff_sent, datetime):
        #         ff_sent = ff_sent.strftime("%Y-%m-%d %H:%M:%S")

        #     processed_data.append({
        #         '_id': row['_id'],
        #         'loanAmount': row.get('loanAmount', 'NA'),
        #         'leadCategory': row.get('leadCategory', 'NA'),
        #         'leadSource': row.get('leadSource', 'NA'),
        #         'leadSegment': row.get('leadSegment', 'NA'),
        #         'lcr.ffSent': ff_sent
        #     })

        logger.info("Got data from JSON file. TOTAL %s", len(processed_data))
        return limited_df.to_dict('records')
        # return processed_data
    except Exception as ex:
        logger.error("Error while reading data from JSON: %s", str(ex))
        return None



#########################recent one new


def get_multi_label_prediction(data_count):
    try:
        logger.info("Started MULTI-LABEL classification")

        # Step 1: Get data   / Process / clean/format data
        dbData = pr.from_db(data_count)
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

    # DONT TOUCH ANYTHING BELOW THIS LINE  COZ NOTHING WORKS HERE PROLLY, SOME WORKS THOUGH

    def get_multi_label_prediction(self, data_count):
        try:
            logger.info("Started MULTI-LABEL classification")

            # Fetch data
            data = get_data.from_db(data_count)

            # Process data
            processed_data = self.process_data(data)
            processed_df = pd.DataFrame(processed_data)

            # Transform labels for multi-label classification
            y, reverse_mapping = self.transform_labels(processed_df)
            print("Class Distribution in y:")
            class_counts = pd.Series(y).value_counts()
            print(class_counts)

            # Check for minimum class size
            min_class_count = class_counts.min()
            if min_class_count < 2:
                logger.error("The least populated class in y has only 1 member. Adjusting data...")
                # Handle underrepresented class (merge, drop, or duplicate samples)
                to_drop = class_counts[class_counts < 2].index.tolist()
                processed_df = processed_df[~processed_df['FF Sent Binary'].isin(to_drop)]
                y = processed_df['FF Sent Binary']
                print("Adjusted Class Distribution:")
                print(pd.Series(y).value_counts())

            # Handle categorical variables
            categorical_columns = ['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']
            processed_df[categorical_columns] = processed_df[categorical_columns].fillna("Unknown").astype(str)

            # Encode categorical variables
            encoded_features = self.encode_categorical_features(processed_df, categorical_columns)
            X = pd.concat([processed_df.drop(columns=categorical_columns), encoded_features], axis=1)

            # Split Data into Train/Test Sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y if min_class_count >= 2 else None
            )

            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            print("Training Class Distribution (Resampled):")
            print(pd.Series(y_train_resampled).value_counts())
            print("Test Class Distribution:")
            print(pd.Series(y_test).value_counts())

            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train_resampled, y_train_resampled)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            print("Reversed Classification Report:")
            y_pred_original = [reverse_mapping[label] for label in y_pred]
            y_test_original = [reverse_mapping[label] for label in y_test]
            print(classification_report(y_test_original, y_pred_original))

            # Confusion Matrix and AUC-ROC
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
            if len(set(y_test)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                print(f"AUC-ROC: {auc:.2f}")

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            return accuracy * 100
        except Exception as ex:
            logger.error(ex)
            return str(ex)

    # Multi-Label Transformation Function
    def get_multi_label_predictionHASERROR(self, data_count):
        try:
            logger.info("Started MULTI-LABEL classification")
            # data = get_data.from_json(data_count)
            data = get_data.from_db(data_count)

            processed_data = self.process_data(data)
            processed_df = pd.DataFrame(processed_data)

            # Transform labels for multi-label classification
            y, reverse_mapping  = self.transform_labels(processed_df)
            # print(y)
            # print(reverse_mapping)

            # Encode categorical variables
            categorical_columns = ['Loan Amount', 'Lead Source', 'Lead Category', 'Lead Segment']
            processed_df[categorical_columns] = processed_df[categorical_columns].fillna("NA").astype(str)
            encoded_features = self.encode_categorical_features(processed_df, categorical_columns)
            X = pd.concat([processed_df.drop(columns=categorical_columns), encoded_features], axis=1)

            # Split Data into Train/Test Sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
            )

            # Handle class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            print("Training Class Distribution (Resampled):")
            print(pd.Series(y_train_resampled).value_counts())
            print("Test Class Distribution:")
            print(pd.Series(y_test).value_counts())

            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train_resampled, y_train_resampled)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            print("Reversed Classification Report:")
            y_pred_original = [reverse_mapping[label] for label in y_pred]
            y_test_original = [reverse_mapping[label] for label in y_test]
            print(classification_report(y_test_original, y_pred_original))

            # Confusion Matrix and AUC-ROC
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion Matrix:\n{cm}")
            if len(set(y_test)) == 2:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                print(f"AUC-ROC: {auc:.2f}")

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            return accuracy * 100
        except Exception as ex:
            logger.error(ex)
            return str(ex)

    def transform_labels(self, data):
        """Transform FF Sent column into binary labels and store mapping."""
        logger.info("Transforming data %s", len(data))
        label_mapping = {"NA": 0, "FF Sent": 1}  # Custom mapping
        data['FF Sent Binary'] = data['FF Sent'].apply(lambda x: label_mapping.get("FF Sent") if x != "NA" else label_mapping.get("NA"))
        return data, {v: k for k, v in label_mapping.items()}  # Return reverse mapping

    def encode_categorical_features(self, df, categorical_columns):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_columns])
        return pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))


    def transform_labels0(data):
        """Transform FF Sent and additional columns into multi-label format."""
        data['MultiLabels'] = data.apply(
            lambda row: ['FF Sent'] if row['FF Sent'] != 'NA' else [], axis=1
        )
        mlb = MultiLabelBinarizer()
        y_multi = mlb.fit_transform(data['MultiLabels'])
        return y_multi, {i: label for i, label in enumerate(mlb.classes_)}

    def get_binary_predictionWORKS(self,data_count):
        try:
            logger.info("Started BINARY classification")

            # Step 1: Get data
            # dbData = getFromDB(data_count)
            dbData = get_data.from_json(data_count)

            #Step 2:  Process / clean / format data
            processed_data = self.process_data(dbData)
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

            # Step 11: Evaluate Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            return accuracy
        except Exception as ex:
            logger.error(ex)
            return str(ex)

"""

        Binary Classification :
            When we need to classify leads into two categories based on whether or not they will reach FFSent,

        Multi -  label Classification:
            When we need to to classify the leads into multiple categories based on whether they will or not reach FFSent





"""
















###########below works ok

class classifierWORKS:

    def get_binary_prediction(self, data_count):
        try:
            logger.info("Started BINARY classification")

            # Step 1: Get data
            # data = get_data.from_db(data_count)
            data = get_data.from_json(data_count)

            #Step 2:  Process / clean / format data
            processed_data = self.process_data(data)
            processed_df = pd.DataFrame(processed_data)
            logger.info("Total PROCESSED data %s",len(processed_df))
            # logger.info("Total PROCESSED data %s",processed_df.to_string(index=False))

            # Step 3: Encode categorical variables
            label_encoder = LabelEncoder()
            categorical_columns = ['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']
            for col in categorical_columns:
                processed_df[col] = label_encoder.fit_transform(processed_df[col].astype(str))


            
            # Step 4:  Prepare features and labels ( x & y) -- FEATURE ENGINEERING
            X = processed_df[['Loan Amount','Lead Source', 'Lead Category', 'Lead Segment']] # predictors , input
            y =  (processed_df['FF Sent'] != "NA").astype(int) # if ff sent has value then convert to 1 else 0. we aint encoding FF send coz its binary classification   // output

            #better training
            wanna_undersample = True
            if wanna_undersample:
                feature, label = self.under_sampling_majority_class(X,y)
                X = feature
                y = label

             # Step 5: Split data into train and test sets [ 80 % training,  20% test] -- SELECTION / INPUT PREPARATION
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

            print("Training Class Distribution:")
            print(y_train.value_counts())
            print("Test Class Distribution:")
            print(y_test.value_counts())

            # Step 6: Ensure X_train and y_train are aligned
            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)

            # Step 7: Feed into Classifier MODEL  : Initialize and train Random Forest Classifier --- Model Training:
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # training features
            model.fit(X_train, y_train) # model training
            dump(model,'binary_classifier_rf_model.joblib')
            logger.info("Model saved as 'binary_classifier_rf_model.joblib'")

            #  Step 8: Make predictions
            y_pred = model.predict(X_test)

            # Step 9: Generate the classification report
            report = classification_report(y_test, y_pred, target_names=['will_not_reach', 'will_reach'])
            print(report)

            # Step 9: Transform the Predicted FFSent (0 or 1)
            actual_predicted_values = ['will_reach' if val == 1 else 'will_not_reach' for val in y_pred]         

            # Step 10: Evaluate Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            return accuracy *100
        except Exception as ex:
            logger.error(ex)
            return str(ex)

     # Clean / Format / Parse data "" to NA

    def process_data(self,records):
        logger.info("Processing records %s", len(records))

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
        logger.info("Processed records %s", len(processed))
        return processed

    def under_sampling_majority_class(self, feature, label ):
        try:
            logger.info("Under Sampling majority class")
            # Combine X and y into a single DataFrame for sampling
            X= feature
            y =label
            data = pd.concat([X, y], axis=1)

            # Separate majority and minority classes
            data_majority = data[data['FF Sent'] == 1]
            data_minority = data[data['FF Sent'] == 0]

            # Undersample the majority class
            data_majority_downsampled = resample(
                data_majority,
                replace=False,  # sample without replacement
                n_samples=1000,  # desired number of majority class samples
                random_state=42
            )

            # Oversample the minority class
            data_minority_upsampled = resample(
                data_minority,
                replace=True,   # sample with replacement
                n_samples=900,  # desired number of minority class samples
                random_state=42
            )

            # Combine minority and majority classes
            balanced_data = pd.concat([data_majority_downsampled, data_minority_upsampled])

            # Separate back into features and labels
            X_balanced = balanced_data.drop('FF Sent', axis=1)
            y_balanced = balanced_data['FF Sent']

            return X_balanced, y_balanced

        except Exception as ex:
            logger.error(ex)
            return str(ex)
