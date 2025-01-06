from sqlite3 import connect
import ssl
from pymongo import MongoClient


class dbConnection:
	def __init__(self,logger,config):
		self.logger = logger 
		self.config = config

	def get_mongo_connection(self,purpose):
		self.logger.info("Getting Mongo DB Connection.")
		self.logger.info(f"Purpose of Db : {purpose}")
		try:
			mongo_db_server = self.config.get_db_server(purpose)
			self.logger.info(f"Connecting to  mongo db server : {mongo_db_server}")
			connection= MongoClient(mongo_db_server)
			self.logger.info("Received Mongo DB Connection.")
			return connection
		except Exception as ex:
			self.logger.error(f"Mongo Database connection failed. Error {ex}")
			return None

	def close_mongo_connection(self, connection):
		try:
			connection.close()
			self.logger.info("Closed Mongo DB Connection.")

		except Exception as ex:
			self.logger.error("Mongo Database connection failed to Close. Error %s", ex)

