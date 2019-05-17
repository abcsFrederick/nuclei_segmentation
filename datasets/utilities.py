def _affix_keyfunc(*affixes):
    def keyfunc(v):
        for prefix, suffix in affixes:
            if v.startswith(prefix) and v.endswith(suffix):
                return v[len(prefix):-len(suffix)]
        return v
    return keyfunc
