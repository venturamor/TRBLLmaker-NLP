import lyricsgenius


class SongInfo:
    def __init__(self, genre, song_genius, annotations: list):
        self.song_genius = song_genius
        self.annotations = annotations
        self.genre = genre
        self.title = getattr(self.song_genius, 'title')

    def get_id(self):
        return getattr(self.song_genius, 'id')

    def get_lyrics(self):
        return getattr(self.song_genius, 'lyrics')



