class Filter:
    def emo(contents):
        if "EMO_UNKNOWN" in contents:
            return False
        else:
            return True