#pragma once

#include <string>
#include <utility>
#include <vector>

// Common interface for every recommendation algorithm.
// UserBasedCF and ItemBasedCF both inherit from this, so the menu
// and evaluator can run either one without knowing which it is.
class Recommender {
public:
    virtual ~Recommender() = default;

    // Predicted rating (1.0–5.0) that userId would give movieId.
    virtual double predict(int userId, int movieId) const = 0;

    // Top k (movieId, predictedRating) pairs for movies the user
    // has NOT rated, sorted from highest to lowest score.
    virtual std::vector<std::pair<int, double>> recommend(int userId, int k) const = 0;

    // Display name, e.g. "User-Based CF".
    virtual std::string name() const = 0;
};