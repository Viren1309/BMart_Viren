from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'Advertising.csv'
MODEL_PATH = BASE_DIR / 'brightmart_model.joblib'


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f'Model saved to: {MODEL_PATH}')
    print(f'R2 on test set: {model.score(X_test, y_test):.4f}')


if __name__ == '__main__':
    train_and_save_model()
