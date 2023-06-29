def recursive_get(di: dict, arr: list):
    if arr:
        return recursive_get(di[arr[0]], arr[1:])
    return di
