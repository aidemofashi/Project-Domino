class Filter:
    def emo(contents):
        if "EMO_UNKNOWN" in contents and len(contents) < 50:
            return False
        else:
            return True