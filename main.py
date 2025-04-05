from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Veri setini oku
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print("\nİlk 5 Satır:\n", df.head())
print("\nVeri Seti Boyutu:", df.shape)
print("\nSütun Bilgileri:\n", df.info())
print("\nEksik Değer Sayısı:\n", df.isnull().sum())
print("\nTekrarlanan Satır Sayısı:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("\nTekrarlanan Satırlar Kaldırıldıktan Sonra Veri Seti Boyutu:", df.shape)

print("\nUyku Bozukluğu Sınıfları ve Sayıları:\n", df['Sleep Disorder'].value_counts())
sns.countplot(data=df, x='Sleep Disorder')
plt.title('Uyku Bozukluğu Sınıf Dağılımı')
plt.show()

# Hedef değişkeni kodlayalım (TÜM VERİ ÜZERİNDE FIT EDİLİYOR)
le = LabelEncoder()
df['Sleep Disorder Encoded'] = le.fit_transform(df['Sleep Disorder'])

# Özellikleri ve hedefi ayır
X = df.drop(columns=['Sleep Disorder', 'Sleep Disorder Encoded'])
y = df['Sleep Disorder Encoded']

# Kategorik ve sayısal sütunları belirle
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

# Ön işleme adımları
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# Ön işleme adımlarını eğitim ve test verisine uygula
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# SMOTE ile dengesizlik giderme
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
print("\nSMOTE Sonrası Sınıf Dağılımı:", Counter(y_train_res))

print("Eğitim Seti Sınıfları:", np.unique(y_train))
print("Test Seti Sınıfları:", np.unique(y_test))
print("LabelEncoder Sınıfları:", le.classes_)

# Modellerin tanımı ve hiperparametre gridleri
model_configs = [
    {
        'name': 'KNN',
        'pipeline': Pipeline([('classifier', KNeighborsClassifier())]),
        'params': {'classifier__n_neighbors': range(1, 21)}
    },
    {
        'name': 'Lojistik Regresyon',
        'pipeline': Pipeline(
            [('classifier', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))]),
        'params': {'classifier__C': np.logspace(-3, 3, 7)}
    },
    {
        'name': 'Karar Ağacı',
        'pipeline': Pipeline([('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))]),
        'params': {
            'classifier__max_depth': range(1, 21),
            'classifier__min_samples_split': range(2, 11),
            'classifier__min_samples_leaf': range(1, 11)
        }
    },
    {
        'name': 'Rastgele Orman',
        'pipeline': Pipeline([('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))]),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
        }
    },
    {
        'name': 'Destek Vektör Makineleri',
        'pipeline': Pipeline([('classifier', SVC(probability=True, class_weight='balanced', random_state=42))]),
        'params': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
    }
]

results = []
for config in model_configs:
    print(f"\n{config['name']} Eğitiliyor...")
    gs = GridSearchCV(config['pipeline'], config['params'], cv=5, scoring='accuracy')
    gs.fit(X_train_res, y_train_res)
    y_pred = gs.predict(X_test_processed)
    y_prob = gs.predict_proba(X_test_processed)

    print(f"\n{config['name']} - En İyi Parametreler: {gs.best_params_}")

    predicted_classes = np.unique(y_pred)
    report_target_names = [str(label) for label in le.inverse_transform(predicted_classes)]  # Açıkça string'e dönüştür

    print(classification_report(y_test, y_pred, target_names=report_target_names, labels=predicted_classes,
                                zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=report_target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{config["name"]} Karmaşıklık Matrisi')
    plt.show()

    results.append({
        'Model': config['name'],
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'ROC-AUC (OVO Macro)': roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
    })

# Sonuçları görselleştir
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by='F1-Score (Macro)', ascending=False)
print("\nModel Performans Özeti:\n", results_df_sorted)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df_sorted, x='Model', y='F1-Score (Macro)')
plt.title('Modellere Göre F1-Score (Macro)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()