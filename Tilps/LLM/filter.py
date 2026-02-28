class Filter:
    @staticmethod
    def emo(contents):
        if ("EMO_UNKNOWN" in contents) and len(contents) < 120:
            return False
        else:
            return True