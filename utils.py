def html2df(wiki, place=0, header=False, index='Country'):
    df = wiki[place]
    if not header:
        df.columns = df.iloc[0]
        df = df.drop(0)
    df = df.set_index(index)
    return df