"""
Load SQL files in the module directory and make them available as strings.
"""


from importlib import resources
from neurobooth_analysis_tools.data import sql


QUERY_CLIN_DEMOG_SCALE = resources.read_text(sql, 'subject_visit_info.sql')
