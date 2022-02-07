import lyricsgenius
from config_parser import config_args
from song_info import SongInfo
from songs_db import SongsInfoDB
from os.path import join, getctime
import glob


def combine_all_gernres(config_args, with_annot=0):
    """
    combine all genres pickle to one uniqe db that contains each song once,
    and only if it has annotations
    :return:
    """
    genres = config_args['data_extraction']['genres'][:-1]
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


if __name__ == '__main__':

    combine_all_gernres(config_args, with_annot=1)