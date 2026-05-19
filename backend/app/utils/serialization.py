def serialize_results(results: list) -> list:
    serialized = []
    for r in results:
        s = {}
        for k, v in r.items():
            if hasattr(v, "tolist"):
                s[k] = v.tolist()
            elif isinstance(v, list):
                s[k] = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            else:
                s[k] = v
        serialized.append(s)
    return serialized