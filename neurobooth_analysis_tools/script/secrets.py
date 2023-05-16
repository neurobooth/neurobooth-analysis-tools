import json
from importlib import resources
from typing import Optional
from neurobooth_analysis_tools import script
from neurobooth_analysis_tools.data.database import DatabaseConnectionInfo, DatabaseConnection


class Secrets:
    def __init__(self, secrets_file: Optional[str] = None):
        """
        Load application/script secrets from a JSON file.
        :param secrets_file: The file to load secrets from. Will load attempt to load a default version if None.
        """
        if secrets_file:
            with open(secrets_file, 'r') as f:
                self.secrets = json.load(f)
        else:
            self.secrets = json.loads(resources.read_text(script, 'secrets.json'))

    def get_database_connection(self) -> DatabaseConnection:
        """
        :return: A database connection object creating using the 'database' key in the secrets dictionary.
        """
        conn_info = DatabaseConnectionInfo(**self.secrets['database'])
        return DatabaseConnection(conn_info)
