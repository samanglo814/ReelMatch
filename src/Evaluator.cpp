
#include "Evaluator.h"

#include "UserBasedCF.h"
#include "ItemBasedCF.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
#include <unordered_map>

using namespace std;

Evaluator::Evaluator(const Dataset& dataset)
    : dataset(dataset) {
}

// Time Complexity: O(T * k), where T is the number of test ratings
// and k is the maximum number of neighbors checked during prediction.
// Calculates the Root Mean Squared Error.
double Evaluator::calculateRMSE(
    const Recommender& recommender,
    const vector<pair<pair<int, int>, float>>& testRatings
) const {

    double sumSquaredErrors = 0;

    for (const auto& test : testRatings) {
        int userId = test.first.first;
        int movieId = test.first.second;
        float actualRating = test.second;

        double predictedRating = recommender.predict(userId, movieId);

        double error = actualRating - predictedRating;
        sumSquaredErrors += error * error;
    }

    if (testRatings.empty()) {
        return 0.0;
    }

    return sqrt(sumSquaredErrors / testRatings.size());
}


// Time Complexity: O(C + 10 * R), where C is the time required
// to generate the top 10 recommendations and R is the number of
// relevant movies. The 10 recommendations are checked against the
// relevant movies using linear search.
// Calculates Precision@10 for a user.
double Evaluator::calculatePrecisionAt10(
    const Recommender& recommender,
    int userId,
    const vector<int>& relevantMovies
) const {

    // Generate the top 10 recommendations.
    auto recommendations = recommender.recommend(userId, 10);

    int relevantFound = 0;

    // Check how many recommendations are relevant.
    for (const auto& recommendation : recommendations) {
        int movieId = recommendation.first;

        if (find(relevantMovies.begin(), relevantMovies.end(), movieId)
            != relevantMovies.end()) {
            relevantFound++;
        }
    }

    return static_cast<double>(relevantFound) / 10.0;
}


// Runs the evaluation and compares both recommendation algorithms.
void Evaluator::run(int userId) const {

    cout << "\n========================================\n";
    cout << "       RECOMMENDATION EVALUATION\n";
    cout << "========================================\n";

    // Copy the original dataset to create the training dataset.
    Dataset trainingDataset = dataset;

    vector<pair<pair<int, int>, float>> testRatings;

    // Create a random number generator.
    mt19937 generator(42);
    uniform_real_distribution<double> distribution(0.0, 1.0);

    // Time Complexity: O(N), where N is the total number of ratings.
    // Split the ratings into 80% training and 20% testing.
    for (const auto& user : dataset.getUserRatings()) {

        int currentUserId = user.first;

        for (const auto& rating : user.second) {

            int movieId = rating.first;
            float actualRating = rating.second;

            if (distribution(generator) < 0.2) {
                testRatings.push_back({
                    {currentUserId, movieId},
                    actualRating
                });

                trainingDataset.removeRating(currentUserId, movieId);
            }
        }
    }

    cout << "\nTraining ratings: " << trainingDataset.numRatings();
    cout << "\nTesting ratings: " << testRatings.size() << "\n";

    // Find relevant movies for every user in the test set.
    // A rating of 4.0 or higher is considered relevant.
    unordered_map<int, vector<int>> relevantMoviesByUser;

    // Time Complexity: O(T), where T is the number of test ratings.
    for (const auto& test : testRatings) {

        int currentUserId = test.first.first;
        int movieId = test.first.second;
        float actualRating = test.second;

        if (actualRating >= 4.0) {
            relevantMoviesByUser[currentUserId].push_back(movieId);
        }
    }

    // Evaluate User-Based Collaborative Filtering.
    cout << "\nEvaluating User-Based CF...\n";

    auto userStart = chrono::high_resolution_clock::now();

    UserBasedCF userBased(trainingDataset);

    auto userEnd = chrono::high_resolution_clock::now();

    double userTrainingTime =
        chrono::duration<double, milli>(userEnd - userStart).count();

    double userRMSE = calculateRMSE(userBased, testRatings);

    // Evaluate Item-Based Collaborative Filtering.
    cout << "Evaluating Item-Based CF...\n";

    auto itemStart = chrono::high_resolution_clock::now();

    ItemBasedCF itemBased(trainingDataset);

    auto itemEnd = chrono::high_resolution_clock::now();

    double itemTrainingTime =
        chrono::duration<double, milli>(itemEnd - itemStart).count();

    double itemRMSE = calculateRMSE(itemBased, testRatings);

    // Select users who have at least one relevant movie in the test set.
    vector<int> evaluationUsers;

    // Time Complexity: O(U), where U is the number of users
    // with at least one relevant movie in the test set.
    for (const auto& [currentUserId, relevantMovies] : relevantMoviesByUser) {
        evaluationUsers.push_back(currentUserId);
    }

    // Sort user IDs before shuffling to ensure a reproducible sample.
    // Time Complexity: O(U log U).
    sort(evaluationUsers.begin(), evaluationUsers.end());

    // Use a fixed seed so the sample is reproducible.
    // Time Complexity: O(U).
    mt19937 precisionGenerator(42);
    shuffle(evaluationUsers.begin(), evaluationUsers.end(), precisionGenerator);

    // Evaluate at most 100 users.
    if (evaluationUsers.size() > 100) {
        evaluationUsers.resize(100);
    }

    // Calculate average Precision@10 for both algorithms.
    double userPrecision = 0.0;
    double itemPrecision = 0.0;

    // Time Complexity: O(E * (P_user + P_item)),
    // where E is the number of evaluated users and P is the
    // time required to generate and check recommendations.
    for (int currentUserId : evaluationUsers) {

        userPrecision += calculatePrecisionAt10(
            userBased,
            currentUserId,
            relevantMoviesByUser.at(currentUserId)
        );

        itemPrecision += calculatePrecisionAt10(
            itemBased,
            currentUserId,
            relevantMoviesByUser.at(currentUserId)
        );
    }

    // Calculate the average Precision@10.
    if (!evaluationUsers.empty()) {
        userPrecision /= evaluationUsers.size();
        itemPrecision /= evaluationUsers.size();
    }

    // Display the results.
    cout << "\n========================================\n";
    cout << "             RESULTS\n";
    cout << "========================================\n";

    cout << fixed << setprecision(4);

    cout << "\nUsers Evaluated: " << evaluationUsers.size() << "\n";

    cout << "\nMetric                 User-Based    Item-Based\n";
    cout << "------------------------------------------------\n";

    cout << "RMSE                   "
         << setw(10) << userRMSE
         << setw(14) << itemRMSE << "\n";

    cout << "Precision@10           "
         << setw(10) << userPrecision
         << setw(14) << itemPrecision << "\n";

    cout << "Training Time (ms)     "
         << setw(10) << userTrainingTime
         << setw(14) << itemTrainingTime << "\n";

    cout << "========================================\n";
}
