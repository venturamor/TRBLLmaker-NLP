import lyricsgenius
from config_parser import config_args
from song_info import SongInfo
from songs_db import SongsInfoDB
import datetime
from os.path import join
from requests.exceptions import Timeout, ConnectionError
import chosen_artists


def get_genre(genre, db_pickle_path=None, page=1):
    """
    gets the lyrics + metadata + annotations of all the songs that have the requested genre tag.
    takes ~0.5 min per song with the following setup.
    :param genre:
    :param db_pickle_path:
    :param page: number
    :return: save pickle
    """
    # # genre
    # per song: full title, url, id, annotation_count, lyrics, (verses, annotations), metadata (genre, artists, album...)
    token = config_args['data_extraction']['token']
    save_every = config_args['data_extraction']['save_songs_db_every']

    genius = lyricsgenius.Genius(token)
    # in order to handle with timeouts
    genius.timeout = 10
    genius.sleep_time = 3
    genius.retries = 5
    retries_num = 2

    name_by_date = genre + '_' + datetime.datetime.today().strftime('%d%m%y_%H%M')   # _%H%M')
    songs_info_db = SongsInfoDB(name=name_by_date, genre=genre, pickle_path=db_pickle_path)

    while page:  # 20 songs at each page
        # returns urls of songs
        res = genius.tag(genre, page=page)
        for hit in res['hits']:
                retries = 0
                while retries < retries_num:
                    try:
                        # lyrics + metadata (by title and main artist)
                        song = genius.search_song(hit['title'], hit['artists'][0])
                        # annotations (by song id)
                        if song == None:
                            retries = retries_num
                            continue
                        else:
                            annotation = genius.song_annotations(song.id)
                            song_info = SongInfo(genre, song, annotation)
                            songs_info_db.add_song(song_info)
                            # save every # songs to pickle
                            if songs_info_db.get_len() % save_every == 0:
                                songs_info_db.save_to_pickle()
                                print('# songs in db:', str(songs_info_db.get_len()))
                                print('current page:', str(page))
                            break  # break to next hit
                    except TimeoutError as e:
                        retries += 1
                        continue
                    except ConnectionError as e:
                        retries += 1
                        songs_info_db.save_to_pickle()
                        print('# songs in db:', str(songs_info_db.get_len()))
                        print('ConnectionError: current page:', str(page))
                        continue

        page = res['next_page']

    # final save
    songs_info_db.save_to_pickle()
    print('# songs in db:', str(songs_info_db.get_len()))
    print('Done: extracting', genre, 'songs :)')
    print('Done')


def all_genres_extraction(config_args):
    """
    main Genius tags - ['country', 'pop', 'r&b', 'rap', 'rock']
    secondary tags (hundreds...) - https://genius.com/Genius-tags-music-genres-international-annotated
    :return:
    """
    genres = config_args['data_extraction']['genres'][:-2]  # without final
    for genre in genres:
        get_genre(genre)


def genre_from_last_point(last_file, genre, page):
    """

    :param last_file:
    :param genre:
    :param page:
    :return:
    """

    db_pickle_path_2_load = join(config_args['data_extraction']['pickles_parent_dir'], genre, last_file)
    get_genre(genre, db_pickle_path_2_load, page=page)


def get_songs_by_artists(chosen_artists, db_pickle_path=None, page=1):
    """
    the genre is artists. no specified genre for each song.
    :param chosen_artist:
    :param db_pickle_path:
    :param page:
    :return:
    """
    token = config_args['data_extraction']['token']
    max_songs_per_artist = config_args['data_extraction']['max_songs_per_artist']
    save_every = config_args['data_extraction']['save_songs_db_every']
    genre = 'artists'
    genius = lyricsgenius.Genius(token)
    # in order to handle with timeouts
    genius.timeout = 10
    genius.sleep_time = 2
    genius.retries = 2
    retries_num = 2

    for ch_artist in chosen_artists:
        name_by_date = ch_artist + '_' + datetime.datetime.today().strftime('%d%m%y_%H%M')   # _%H%M')
        songs_info_db = SongsInfoDB(name=name_by_date, genre=genre, pickle_path=db_pickle_path)
        artist = genius.search_artist(ch_artist, max_songs=max_songs_per_artist)
        if artist == None:
            print('Warning: The artist:', ch_artist, 'was not found.')
            continue
        if len(artist.songs) < 3:  # not enough songs -> no worthy
            continue

        for song in artist.songs:
            retries = 0
            while retries < retries_num:
                try:
                    annotation = genius.song_annotations(song.id)
                    song_info = SongInfo(genre, song, annotation)
                    songs_info_db.add_song(song_info)
                    # save every # songs to pickle
                    if songs_info_db.get_len() % save_every == 0:
                        songs_info_db.save_to_pickle(pi_name=ch_artist)
                        print('# songs in db:', str(songs_info_db.get_len()))
                    break  # break to next hit

                except TimeoutError as e:
                    retries += 1
                    continue
                except ConnectionError as e:
                    retries += 1
                    songs_info_db.save_to_pickle(pi_name=ch_artist)
                    print('# songs in db:', str(songs_info_db.get_len()))
                    continue


        # save when finish artists songs
        songs_info_db.save_to_pickle(pi_name=ch_artist)
        print('# songs in db:', str(songs_info_db.get_len()))
        continue

    # final save
    songs_info_db.save_to_pickle(pi_name=ch_artist)
    print('# songs in db:', str(songs_info_db.get_len()))
    print('Done: extracting', genre, 'songs :)')
    print('Done')


if __name__ == '__main__':
    # # one genre extraction
    # genre = 'country'
    # get_genre(genre)
    #
    # # genre from last checkpoint
    # last_file = 'rap_050222_1022.pickle'
    # page = 43
    # genre_from_last_point(last_file, genre, page)
    #
    # # all genres extraction
    # all_genres_extraction(config_args)
    #
    # by artists
    chosen_artists = chosen_artists.chosen_artists
    get_songs_by_artists(chosen_artists, db_pickle_path=None, page=1)


# default search by title: A-Z/ pageviews / release date

# # # artist search
# artist = genius.search_artist("Imagine Dragons", max_songs=1, sort="popularity", include_features=True)
# print(artist.songs)
# #
# # # song by artist and song's title
# # has song_id inside
# song = genius.search_song("enemy", artist.name)
# print(song.lyrics)
# #
# # # annotations
# # a = genius.song_annotations(5992642) #song id