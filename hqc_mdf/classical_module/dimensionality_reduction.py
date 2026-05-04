from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_tsne(X, n_components=2):
    """
    Applies t-SNE for dimensionality reduction/visualization.
    """
    tsne = TSNE(n_components=n_components)
    X_tsne = tsne.fit_transform(X)
    return X_tsne, tsne

def apply_lda(X_train, X_test, y_train, n_components=1):
    """
    Applies LDA for dimensionality reduction.
    """
    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda, lda
