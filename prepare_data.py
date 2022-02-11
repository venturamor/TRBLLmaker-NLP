import lyricsgenius
from song_info import SongInfo
from songs_db import SongsInfoDB
from config_parser import config_args
import pickle
import pandas as pd
from os.path import isfile, join, exists


def create_data_samples(db_pickle_path):

    """
    load db pickle and creates :
    - samples df - title, artist, song_id, text (sentence), annotation (meaning)
    - songs df - title, artist, genre, song_genius columns, lyrics
    songs_list: [SongInfo]: {annotations: [tuple : (str, list - annotation)],
                             genre: str,
                             title: str
                             song_genius: Song: {artist, id, lyrics, lyrics_state,
                                                 stats: {hot - bool, pageviews},
                                                 pyongs_count, url, image_url}}
     """

    # load pickle into songsInfoDB
    genre = 'final'
    songs_info_db = SongsInfoDB(name='for_dataset', genre=genre, pickle_path=db_pickle_path)
    # create dfs
    samples_df = pd.DataFrame(columns=['title', 'artist', 'song_id', 'text', 'annotation'])
    songs_df = pd.DataFrame()

    for song in songs_info_db.songs_list:
    # samples_df
        for annot in song.annotations:
            row_sample = {'title': song.title, 'artist': song.song_genius.artist, }


    print('done')


if __name__ == '__main__':
    # for creating train, test, validation data jsons
    pickle_dir = config_args["data_extraction"]["pickles_parent_dir"]
    genre = 'final'
    pickle_name = 'final_070222_0951.pickle'
    db_pickle_path = join(pickle_dir, genre, pickle_name)
    create_data_samples(db_pickle_path)