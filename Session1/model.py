from sklearn.ensemble import RandomForestClassifier


def build_model() -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,  # number of trees in the forest
        max_depth=10,  # number of levels in the tree
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
        verbose=1,  # output progress
    )
    return model