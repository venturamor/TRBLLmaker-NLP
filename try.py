import lyricsgenius




token = 'hUi9AV743dOI_hVP2CBayG51voTzJjCCToU9ef3NnVV9FbpomnY35F2D9ygeMZ9X'
genius = lyricsgenius.Genius(token)
# default search by title: A-Z/ pageviews / release date
# artist = genius.search_artist("Imagined Dragons", max_songs=10, sort="popularity")
# print(artist.songs)

artist = genius.search_artist("Imagine Dragons", max_songs=3, sort="popularity", include_features=True)
print(artist.songs)

song = genius.search_song("enemy", artist.name)
print(song.lyrics)

# annotations
a = genius.song_annotations(5992642) #song id

# # genre
# # this gets the lyrics of all the songs that have the pop tag.
# page = 1
# lyrics = []
# while page:
#     res = genius.tag('pop', page=page)
#     for hit in res['hits']:
#         song_lyrics = genius.lyrics(song_url=hit['url'])
#         lyrics.append(song_lyrics)
#     page = res['next_page']

