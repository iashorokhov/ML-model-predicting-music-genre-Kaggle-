import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

class MusicGenreClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['acousticness', 'danceability', 'duration_ms', 'energy',
                                           'instrumentalness', 'liveness', 'loudness', 'speechiness',
                                           'tempo', 'valence']),
                ('cat', OneHotEncoder(sparse_output=False), ['key', 'mode'])
            ]
        )
        self.pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        self.parameters = {
            'classifier__n_estimators': [10, 30, 50],
            'classifier__max_depth': [10, 20]
        }
        self.gridsearch = GridSearchCV(self.pipeline, self.parameters, cv=5)
        self.smote = SMOTE()
        
    def fit(self, X, y):
        y_encoded = self.le.fit_transform(y)
        X_processed = self.preprocessor.fit_transform(X)
        X_resampled, y_resampled = self.smote.fit_resample(X_processed, y_encoded)
        self.gridsearch.fit(X_resampled, y_resampled)
        
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.gridsearch.predict(X_processed)
    
    def predict_and_decode(self, X):
        y_pred_encoded = self.predict(X)
        return self.le.inverse_transform(y_pred_encoded)

# Загрузка данных
data = pd.read_csv('kaggle_music_genre_train.csv')
data_test = pd.read_csv('kaggle_music_genre_test.csv')

# Удаляем пропуски и лишние столбцы, преобразуем типы столбцов
data = (data.dropna()
            .drop(['obtained_date', 'track_name'], axis=1)
            .astype({'key': 'category', 'mode': 'category'}))

data_test = (data_test.dropna()
                     .drop(['obtained_date', 'track_name'], axis=1)
                     .astype({'key': 'category', 'mode': 'category', 'instance_id': 'float64'}))

# Выделение признаков и целевого признака
X = data.drop('music_genre', axis=1)
y = data['music_genre']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание экземпляра классификатора
classifier = MusicGenreClassifier()

# Обучение модели
classifier.fit(X_train, y_train)

# Прогнозы на тестовой выборке
y_pred = classifier.predict(X_test)

# Предсказания для новых данных
y_pred_new = classifier.predict_and_decode(data_test)

# Оформляем предсказания в датафрейм
predictions = pd.DataFrame({'instance_id': data_test['instance_id'], 'music_genre': y_pred_new})

# Сохраняем предсказания в файл
predictions.to_csv('predictions.csv', index=False)