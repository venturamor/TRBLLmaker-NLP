import lyricsgenius
from song_info import SongInfo
from songs_db import SongsInfoDB
from config_parser import config_args
import pickle
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import isfile, join, exists
from os import listdir, walk, makedirs



def pickle_2_dataframes(db_pickle_path):

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
    songs_df = pd.DataFrame(columns=['title', 'artist', 'song_id', 'genre', 'lyrics', 'lyrics_state',
                                     'stat_hot', 'stat_pageviews', 'url', 'image_url'])
    idx_annot = 0
    for idx, song in enumerate(songs_info_db.songs_list):
        #  songs_df
        try:
            pageviews = song.song_genius.stats.pageviews
        except:
            pageviews = None
        row_song = {'title': song.title, 'artist': song.song_genius.artist,
                    'song_id': song.song_genius.id, 'genre': song.genre,
                    'lyrics': song.song_genius.lyrics, 'lyrics_state': song.song_genius.lyrics_state,
                    'stat_hot': song.song_genius.stats.hot,
                    'stat_pageviews': pageviews,
                    'url': song.song_genius.url, 'image_url': song.song_genius.header_image_url}
        songs_df = songs_df.append(pd.DataFrame(row_song, index=[idx]))

        #  samples_df
        for annot in song.annotations:
            row_sample = {'title': song.title, 'artist': song.song_genius.artist,
                          'song_id': song.song_genius.id,
                          'text': annot[0], 'annotation': annot[1][0][0]}
            samples_df = samples_df.append(pd.DataFrame(row_sample, index=[idx_annot]))
            idx_annot += 1

    # save to json
    jsons_dir = config_args['data_extraction']['jsons_dir']
    if not exists(jsons_dir):
        makedirs(jsons_dir)
    json_ext = '.json'
    samples_by_date = 'samples' + '_' + datetime.datetime.today().strftime('%d%m%y_%H%M') + json_ext
    songs_by_date = 'songs' + '_' + datetime.datetime.today().strftime('%d%m%y_%H%M') + json_ext

    samples_df.to_json(join(jsons_dir, samples_by_date))
    songs_df.to_json(join(jsons_dir, songs_by_date))

    print('done creating jsons out of pickles')


def split_by_songs(songs_json_path, samples_json_path):
    """
    splits songs json and samples json to train, test and validation.
    samples are splitted by songs split (song id).
    :param take_mini:
    :param songs_json_path:
    :param samples_json_path:
    :return: save new jsons (train, test, validation) per songs and samples
    """

    test_size = config_args["train_args"]["test_size"]
    take_mini = config_args["train_args"]["take_mini"]

    songs_df = pd.read_json(songs_json_path)
    samples_df = pd.read_json(samples_json_path)

    # Leave only 10% of songs. This is a mini dataset.
    if take_mini != 0:  # if different from 0
        all_the_rest, songs_df = train_test_split(songs_df, test_size=take_mini)

    # songs df split
    train_songs, test_songs = train_test_split(songs_df, test_size=test_size)

    # keep same songs split to samples split
    train_songs_ids = train_songs['song_id']
    test_songs_ids = test_songs['song_id']

    train_samps = samples_df[samples_df['song_id'].isin(list(train_songs_ids.values))]
    test_samps = samples_df[samples_df['song_id'].isin(list(test_songs_ids.values))]

    test_size_samps = test_samps.shape[0] / samples_df.shape[0]
    print('Notice: samples test size is -', test_size_samps, '\n',
          '\t\twhile requested songs test size is - ', test_size)

    # split to validation
    train_songs, validation_songs = train_test_split(train_songs, test_size=test_size)
    train_samps, validation_samps = train_test_split(train_samps, test_size=test_size)

    # save the dataframes to a json file
    data_dir = config_args['train_args']['data_dir']
    if not exists(data_dir):
        makedirs(data_dir)

    str_parts = config_args['train_args']['parts']

    if take_mini != 0:
        str_parts = [part + '_mini' for part in str_parts]

    dirs = config_args['train_args']['data_type']
    for dir in dirs:
        if not exists(join(data_dir, dir)):
            makedirs(join(data_dir, dir))

    songs = [train_songs, test_songs, validation_songs]
    samps = [train_samps, test_samps, validation_samps]
    for idx in range(len(songs)):

        songs[idx].to_json(join(data_dir, dirs[0], str_parts[idx] + '.json'))
        samps[idx].to_json(join(data_dir, dirs[1], str_parts[idx] + '.json'))

    print('done')


if __name__ == '__main__':
    # for creating train, test, validation data jsons
    pickle_dir = config_args["data_extraction"]["pickles_parent_dir"]
    genre = 'final'
    pickle_name = 'final.pickle'
    db_pickle_path = join(pickle_dir, genre, pickle_name)
    # pickle_2_dataframes(db_pickle_path)

    # split songs
    json_song = 'songs_cleaned.json'  # 'songs_final.json'
    json_samp = 'samples_cleaned.json'  # 'samples_final.json'
    songs_json_path = join(config_args['data_extraction']['jsons_dir'], json_song)
    samples_json_path = join(config_args['data_extraction']['jsons_dir'], json_samp)
    split_by_songs(songs_json_path, samples_json_path)
    print('done')