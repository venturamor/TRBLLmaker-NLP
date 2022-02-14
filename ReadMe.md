### TRBLLmaker - ReadMe

######  Genius API:
- API client created by https:
//genius.com/api-clients/new
Songify

- app website:
http://example.com/
- Client ID:
PGt8xmgp20zvFyDZb7Q0BtL4VHVLnXWrRvp_4epOwaQ_lI9wZFtDbkY38dAdo_Gi
Client Key (secret):
SMId2JId-TisFzgZKT2z3yOuxIa0OgrsN4V6nYqFfWji66eMl2bAF0LbGo1TLx7CW0vYybPlc2gm2pumRTrgkQ
- Client Access TOKEN:
hUi9AV743dOI_hVP2CBayG51voTzJjCCToU9ef3NnVV9FbpomnY35F2D9ygeMZ9X

- relevant helpful websites:
    - https://docs.genius.com/#/getting-started-h1
    - https://ewenme.github.io/geniusr/articles/geniusr.html
    - https://pypi.org/project/lyricsgenius/
    - https://github.com/ArinkB/Predicting-Song-Skips/blob/master/1_Data%20Acquisition.ipynb
    
######  Data Extraction and preparation:
- [data_extraction](data_extraction.py) - extract songs data and metadata using genius API, 
  by genre and by artists [chosen_artists](chosen_artists.py). saved in pickles (db_pickles/artist or db_pickles/genre)
- [data_arrangement](data_arrangement.py) - gather all extracted data to uniq set (db_pickles/final)
- [prepare_data](prepare_data.py) - organize data in dataframe format (./jsons), 
  and split it to train, test and validation (./data)
  
###### Data Exploration
- Ideas:
 - 