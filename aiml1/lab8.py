class NaiveBayes:
    def __init__(self):
        self.classes = set()
        self.priors = {}
        self.conditional_probs = {}
        self.feature_values = {}

    def fit(self, X, y):
        self.classes = set(y)
        total_samples = len(y)

        # Calculate prior probabilities
        for c in self.classes:
            self.priors[c] = y.count(c) / total_samples

        # Find possible values for each feature
        for i in range(len(X[0])):
            self.feature_values[i] = set(row[i] for row in X)

        # Calculate conditional probabilities with Laplace smoothing
        for c in self.classes:
            self.conditional_probs[c] = {}

            class_samples = [
                X[i] for i in range(len(X))
                if y[i] == c
            ]

            for feature_index in range(len(X[0])):
                self.conditional_probs[c][feature_index] = {}

                for value in self.feature_values[feature_index]:

                    count = sum(
                        1 for row in class_samples
                        if row[feature_index] == value
                    )

                    probability = (
                        count + 1
                    ) / (
                        len(class_samples)
                        + len(self.feature_values[feature_index])
                    )

                    self.conditional_probs[c][feature_index][value] = probability

    def predict(self, x):
        probabilities = {}

        for c in self.classes:

            probability = self.priors[c]

            for feature_index, value in enumerate(x):

                probability *= self.conditional_probs[c][
                    feature_index
                ].get(value, 1 / (
                    len(self.feature_values[feature_index])
                    + 1
                ))

            probabilities[c] = probability

        return max(
            probabilities,
            key=probabilities.get
        )


if __name__ == "__main__":

    dataset = [
        ['Sunny', 'High', 'No'],
        ['Sunny', 'High', 'No'],
        ['Overcast', 'High', 'Yes'],
        ['Rainy', 'Normal', 'Yes'],
        ['Rainy', 'Normal', 'Yes']
    ]

    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    model = NaiveBayes()
    model.fit(X, y)

    test_instance = ['Sunny', 'Normal']

    prediction = model.predict(test_instance)

    print("Test Instance:", test_instance)
    print("Predicted Class:", prediction)