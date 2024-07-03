# Import 
from NHL_gameData_process_objects import *

# Print current directory for validation
print(dir_data)

yearcall = "2023"

# Pull data from NHL - API
obj_apicall = nhl_api_datapull(year=yearcall, dir_data=dir_data)
obj_apicall.datacall(postseason=True) # Initiate call of pickle data

# Initial data process
#   Need to process regular and post season separately
obj_dataproc_baseline = nhl_dataproc_baseline(year=yearcall, dir_data=dir_data)
obj_dataproc_baseline.dataproc(idx_seasontype='02')
obj_dataproc_baseline.dataproc(idx_seasontype='03')


# Create team metric measurements
#   Need to process regular and post season separately
obj_dataproc_eval_teamsuccess = nhl_dataproc_teamsuccess(year=yearcall, dir_data=dir_data)
obj_dataproc_eval_teamsuccess.dataproc(idx_seasontype='02')
obj_dataproc_eval_teamsuccess.dataproc(idx_seasontype='03')

# Create data sequence for play-by-play
obj_dataproc_sequence = nhl_dataproc_sequence(year=yearcall, dir_data=dir_data)
obj_dataproc_eval_teamsuccess.dataproc(idx_seasontype='02')
obj_dataproc_eval_teamsuccess.dataproc(idx_seasontype='03')