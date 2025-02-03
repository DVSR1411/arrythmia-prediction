from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer

def train_stroke_prediction_model(df, x_column="id", y_column="stroke"):
    def apply_results(label):
        if (label == 0):
            return 0  # No Risk
        elif (label == 1):
            return 1  # More Risk

    df['results'] = df[y_column].apply(apply_results)
    
    x = df[x_column]
    y = df['results']

    cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
    x = cv.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    
    models = []
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models.append(('naive_bayes', nb))

    # SVM
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    models.append(('svm', svm))

    # Logistic Regression
    lr = LogisticRegression(random_state=0, solver='lbfgs')
    lr.fit(X_train, y_train)
    models.append(('logistic', lr))

    # Decision Tree
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    # Ensemble classifier
    classifier = VotingClassifier(models)
    classifier.fit(X_train, y_train)
    
    return classifier, cv