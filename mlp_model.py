from sklearn.neural_network import MLPClassifier


# TODO: Use tensorflow instead of SKLearn
def create_mlp_model():
    """
    Replace the torch MLPModel with sklearn's MLPClassifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    return model
