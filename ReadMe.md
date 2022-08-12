### TRBLLmaker - ReadMe

######  Genius API:
- API client created by https:
//genius.com/api-clients/new
Songify

- app website:
http://example.com/
- Client ID:

Client Key (secret):

- Client Access TOKEN:


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

##### Dataset
Working with HuggingFace Dataset format.
- [TRBLL_dataset](TRBLL_dataset.py) - our Dataset struct - takes the jsons that are located in ./data, by config - train_args.
- Dataset include train, test and validation DatasetDicts.
  
###### Data Exploration
- Before splitting to train, test and validation, we can:
    - Print statistics of songs by length, genre, artist, etc.
    - Words cloud of songs lyrics.
    - Words cloud of sentences in songs lyrics that is annotated.
    - Words cloud of the annotated sentences.
    - Statistics from the zero-shot.
    - Correlation between page ranking and other features.
- After splitting to train, test and validation, we can:
    - Print out several sentences with annotations
    - Print statistics of sentences with annotations by length (both song and annotation)

After looking at the data:
- We can see that a lot of the annotations has the artist name in it.
- Some annotations rely on previous songs.
- Some annotations rely on the full lyrics.
- Some annotations have noise like:
  - https
- Some songs are in other languages (Russion, Espanol, French)

##### Future work:
  - Insert a paragraph and annotation to a model and get the sentence that the annotation is talking about.
  - Insert a paragraph and a sentence to a model and get the annotation that the sentence is talking about.
  - Insert a paragraph and information about the artist and get the sentence that the annotation is talking about.

##### Problems:
  - The annotations have a lot of names and history of the artists.
    - Solutions:
       - NER (named entity recognition) and replace the names with some generic words.
       - Remove examples with names.
       - insert the name of the artist with the sentence.
