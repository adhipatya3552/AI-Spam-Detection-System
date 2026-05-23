import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')

    df.columns = ['label', 'text']

    df['label'] = df['label'].map({
        'ham': 0,
        'spam': 1
    })

    return df