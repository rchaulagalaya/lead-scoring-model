from enum import Enum
class DataSource(Enum):
    JSON = "JSON"
    DATABASE = "DATABASE"

class ClassifierType(Enum):
    BINARY = "BINARY"
    NON_BINARY = "NON_BINARY"