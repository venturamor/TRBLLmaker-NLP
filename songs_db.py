from song_info import SongInfo
import pickle
from os import listdir, walk, makedirs
from os.path import isfile, join, exists
from config_parser import config_args
import datetime

# -----------------------------------------------------


class SongsInfoDB:
    genres = config_args['data_extraction']['genres']
    pickles_parent_dir = config_args['data_extraction']['pickles_parent_dir']

    def __init__(self, name: str, genre: str, pickle_path=None):
        """
        takes the new name even if it based on existing db (required for saving new pickle..)
        :param genre: DB type by genre
        :param name: instance (DB) name = pickle name
        :param pickle_path: DB already exist - load it - and update it with new songs
        """
        assert(genre in self.genres)
        self.pickle_path = pickle_path  # origin pickle
        self.genre = genre
        # load pickle
        if self.pickle_path is not None:
            self.load_from_pickle()
        else:
            self.songs_list = []

        self.name = name  # keep the new name if its new (required for saving new pickle..)

    def load_from_pickle(self):
        """
        load db (list_trans) from exist pickle
        """

        print('load db from pickle')
        with open(self.pickle_path, 'rb') as input_pickle:
            db_from_pickle = pickle.load(input_pickle)

        self.songs_list = db_from_pickle.songs_list
        if self.genre != db_from_pickle.genre:
            print('NOTICE: db_type (genre) of pickle is different.\n'
                  'applying pickles db_type')
            self.genre = db_from_pickle.genre  # take the type from loaded
        del db_from_pickle

    def save_to_pickle(self):
        """
        save current db to a pickle
        """
        pickle_ext = '.pickle'
        # pickle_name = self.name + pickle_ext
        pickle_name = self.genre + '_' + datetime.datetime.today().strftime('%d%m%y_%H%M') + pickle_ext
        print('pickling ', self.name,  'to: ', pickle_name)
        # create pickles dir if not exists
        pickles_subdir = join(self.pickles_parent_dir, self.genre)
        if not exists(pickles_subdir):
            makedirs(pickles_subdir)

        # create pickle for this db (variable)
        pickle_path = join(pickles_subdir, pickle_name)
        with open(pickle_path, "wb") as output_pickle:
            pickle.dump(self, output_pickle)

    def song_already_exist(self, new_song: SongInfo):
        """
        checks if this song already in this DB (by id)
        :param new_song:
        :return: boolean - true: if already exist in this db
        """

        # check equality
        new_song_id = new_song.get_id()
        if any([new_song_id == song.get_id() for song in self.songs_list]):
            return True

        return False

    def add_song(self, new_song: SongInfo):
        """
        add new song to this db
        :param new_song:
        :return:
        """
        if not self.song_already_exist(new_song):
            self.songs_list.append(new_song)
        else:
            print('Song', new_song.title,  'already in this', self.name, ' DB')

    def get_len(self):
        return len(self.songs_list)