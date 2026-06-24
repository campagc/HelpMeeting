class Transcript:
    def __init__(self):
        self._buffer = []
        self._pointer = 0

    def append(self, text):
        self._buffer.append(text)

    def take_delta(self):
        delta = "".join(self._buffer[self._pointer:])
        self._pointer = len(self._buffer)
        return delta
