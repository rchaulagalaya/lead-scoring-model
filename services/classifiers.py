import pandas as pd
import logging

# from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from xml.etree.ElementInclude import include
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE
from sklearn.utils import _param_validation

from services import mongo_service as mongoDbService
from services import processors as processorService
# set logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Mongo db
db = mongoDbService.dbConnection()
get_data = processorService.data_service()

class classifier:

    def get_binary_prediction(self, data_count):
        try:
            logger.info("Started BINARY classification")

            # Step 1: Get data
            data = get_data.from_db(data_count)
            # data = get_data.from_json(data_count)

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
            wanna_undersample = False
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

            #  Step 8: Make predictions
            y_pred = model.predict(X_test)

            # Step 9: Generate the classification report
            report = classification_report(y_test, y_pred, target_names=['will_not_reach', 'will_reach'])
            print(report)

            # Step 9: Transform the Predicted FFSent (0 or 1)
            actual_predicted_values = ['will_reach' if val == 1 else 'will_not_reach' for val in y_pred]
            # print("Actual Predicted Values:")
            # print(actual_predicted_values)

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






    # DONT TOUCH ANYTHING BELOW THIS LINE  COZ NOTHING WORKS HERE

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
