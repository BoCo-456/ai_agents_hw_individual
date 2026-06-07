import pandas as pd

if __name__ == '__main__':
    num_validation = 100

    df = pd.read_csv('medium-english-50mb.csv')
    val = df.sample(n=num_validation)
    rest = df.drop(val.index)

    val.to_csv('val.csv', index=False)
    print(val)
    rest.to_csv('rest.csv', index=False)