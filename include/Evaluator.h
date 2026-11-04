#pragma once

#include "Dataset.h"
#include "Recommender.h"

#include <vector>
#include <utility>

class Evaluator {
public:
    // Creates an evaluator using the full dataset
    Evaluator(const Dataset& dataset);

    // Runs the evaluation and compares both recommendation algorithms
    void run(int userId) const;

    // Compares the training times of both algorithms using different dataset sizes
    void runScalingBenchmark() const;

private:
    const Dataset& dataset;

    // Calculates the Root Mean Squared Error
    double calculateRMSE(
        const Recommender& recommender,
        const std::vector<std::pair<std::pair<int, int>, float>>& testRatings
    ) const;

    // Calculates Precision@10 for a user
    double calculatePrecisionAt10(
        const Recommender& recommender,
        int userId,
        const std::vector<int>& relevantMovies
    ) const;
};