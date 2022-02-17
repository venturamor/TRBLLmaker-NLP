import lyricsgenius
from config_parser import config_args
from song_info import SongInfo
from songs_db import SongsInfoDB
from os.path import join, getctime
import os
import glob


def combine_all_gernres(config_args, with_annot=0):
    """
    combine all genres pickle to one uniqe db that contains each song once,
    and only if it has annotations
    :return:
    """
    genres = config_args['data_extraction']['genres'][:-2]
    pickles_dir = config_args['data_extraction']['pickles_parent_dir']
    all_songs_db = SongsInfoDB(name='all_songs_db', genre='final')
    file_type = '\*pickle'

    for genre in genres:
        # get latest pickle in genre
        print('Start adding', genre, 'songs.')
        genre_path = join(config_args['data_extraction']['pickles_parent_dir'], genre)
        files = glob.glob(genre_path + file_type)
        last_file = max(files, key=getctime)
        # load it
        genre_db = SongsInfoDB(name=genre, genre=genre, pickle_path=last_file)
        # add songs
        for song_info in genre_db.songs_list:
            if (with_annot and len(song_info.annotations)) or not with_annot:
                all_songs_db.add_song(song_info)
            else:
                continue

        print(genre, 'songs were added.\n', 'Currently, final db contains:', str(all_songs_db.get_len()))

    all_songs_db.save_to_pickle()
    print('Final db by genres is ready!')


def combine_all_artists(config_args, with_annot=0):
    """
    combine all artists pickle to one uniqe db that contains each song once,
    and only if it has annotations
    :return:
    """
    genre = 'artists'
    pickles_dir = config_args['data_extraction']['pickles_parent_dir']
    all_songs_db = SongsInfoDB(name='all_songs_db', genre='final')
    file_type = '\*pickle'
    for root, dirs, files in os.walk(join(pickles_dir, genre)):
        for file in files:
            file_path = join(root, file)
            artist_db = SongsInfoDB(name=genre, genre=genre, pickle_path=file_path)
            # add songs
            for song_info in artist_db.songs_list:
                if (with_annot and len(song_info.annotations)) or not with_annot:
                    all_songs_db.add_song(song_info)
                else:
                    continue

        print(genre, 'songs were added.\n', 'Currently, final db contains:', str(all_songs_db.get_len()))

    all_songs_db.save_to_pickle()
    print('Final db by artists is ready!')

def combine_final(config_args, genres_final_pickle, artists_final_pickle, with_annot=0):
    """
    combine all genres pickle to one uniqe db that contains each song once,
    and only if it has annotations
    :return:
    """
    genre = 'final'
    # all songs gets all artists final
    all_songs_db = SongsInfoDB(name='all_songs_final_db', genre='final', pickle_path=artists_final_pickle)
    genres_db = SongsInfoDB(name=genre, genre=genre, pickle_path=genres_final_pickle)
    # add songs of genres
    for song_info in genres_db.songs_list:
        if (with_annot and len(song_info.annotations)) or not with_annot:
            all_songs_db.add_song(song_info)
        else:
            continue

        print(genre, 'songs were added.\n', 'Currently, final db contains:', str(all_songs_db.get_len()))

    all_songs_db.save_to_pickle()
    print('Final db is ready to rock!')


if __name__ == '__main__':

    # combine_all_gernres(config_args, with_annot=1)
    # combine_all_artists(config_args, with_annot=1)

    genre = 'final'
    pickles_dir = config_args['data_extraction']['pickles_parent_dir']
    artists_final_pickle = join(pickles_dir, genre, 'final_140222_2003.pickle')
    genres_final_pickle = join(pickles_dir, genre, 'final_070222_0951.pickle')
    combine_final(config_args, genres_final_pickle, artists_final_pickle, with_annot=0)