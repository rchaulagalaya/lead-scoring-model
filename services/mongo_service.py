import sys, os
import logging
from pymongo import MongoClient
# set logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class dbConnection:
	def get_mongo_connection(self):
		logger.info("Getting Mongo DB Connection.")

		username ="mongouser"
		password="H8zYgzuQbTb5psyc"

		try:
			# mongo_server = "mongodb+srv://{username}:{password}@cluster0-dk5qf.mongodb.net/"
			mongo_server = "mongodb+srv://mongouser:H8zYgzuQbTb5psyc@cluster0-dk5qf.mongodb.net/" # dev server
			# mongo_server = "mongodb://localhost:27017/"
			# mongo_server = "mongodb+srv://mongouser:mPtojUhz4TadduZ6@cluster0-8zgt8.mongodb.net/test?retryWrites=true&w=majority" # prod server

			logger.info("mongo_server : ")
			logger.info(mongo_server)

			connection = MongoClient(mongo_server.format(username=username, password=password))

			# logger.info("mongo connection  : ")
			logger.info("Received Mongo DB Connection.")

			return connection
		except Exception as ex:
			logger.error(f"Mongo Database connection failed. Error {ex}")

	def close_mongo_connection(self, connection):
		try:
			connection.close()
			logger.info("Closed Mongo DB Connection.")

		except Exception as ex:
			logger.error("Mongo Database connection failed to Close. Error %s", ex)

