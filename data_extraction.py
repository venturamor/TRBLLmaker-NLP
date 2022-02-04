import lyricsgenius
from config_parser import config_args
from song_info import SongInfo
from songs_db import SongsInfoDB
import datetime
from os.path import join
from requests.exceptions import Timeout


def get_genre(genre, db_pickle_path=None, page=1):
    # # genre
    # this gets the lyrics of all the songs that have the pop tag.
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

        page = res['next_page']

    # final save
    songs_info_db.save_to_pickle()
    print('# songs in db:', str(songs_info_db.get_len()))
    print('current page:', str(page))
    print('Done: extracting', genre, 'songs :)')

if __name__ == '__main__':
    genre = 'pop'
    last_file = 'pop_040222_1555.pickle'
    db_pickle_path_2_load = join(config_args['data_extraction']['pickles_parent_dir'], genre, last_file)
    page = 31
    get_genre(genre, db_pickle_path_2_load, page=page)

# main Genius tags - ['country', 'pop', 'r&b', 'rap', 'rock']
# secondary tags (hundreds...) - https://genius.com/Genius-tags-music-genres-international-annotated


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