class TranscribeResult:
    """
    Each TranscribeResult object represents one SRT line.
    """

    def __init__(self, text: str, start_time: float, end_time: float):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return f"TranscribeResult(text={self.text}, start_time={self.start_time}, end_time={self.end_time})"

    def __repr__(self):
        return str(self)
