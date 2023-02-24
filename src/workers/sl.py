"""
Encapsulates worker for training model on pro-player data
"""

def start():
    return SLWorker.start()

class SLWorker:
    def __init__(self):
        return NotImplemented

    def start(self):
        """
        Start actual training the SL model
        """
        return NotImplemented

    def get_games(self):
        """
        Get all games from SL game directory
        """
        return NotImplemented

def get_game():
    """
    Get all games from a PGN game file
    """
    return NotImplemented