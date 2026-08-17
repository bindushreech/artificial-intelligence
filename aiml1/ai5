def find_s(training_data):

    # Initialize hypothesis with the first positive example
    hypothesis = None

    for row in training_data:
        if row[-1] == 'Yes':
            hypothesis = row[:-1].copy()
            break

    # If there are no positive examples
    if hypothesis is None:
        return "No positive instances found."

    # Update hypothesis using remaining positive examples
    for row in training_data:

        if row[-1] == 'Yes':

            for i in range(len(hypothesis)):

                if row[i] != hypothesis[i]:
                    hypothesis[i] = '?'

    return hypothesis


# Example Usage
if __name__ == "__main__":

    # Columns:
    # Sky, Temp, Humidity, Wind, Water, Forecast, EnjoySport

    dataset = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]

    most_specific_hypothesis = find_s(dataset)

    print(
        "Most Specific Hypothesis found by Find-S:",
        most_specific_hypothesis
    )