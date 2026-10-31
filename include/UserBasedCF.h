#pragma once

#include "Dataset.h"
#include "Recommender.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// User-Based Collaborative Filtering (Resnick et al., 1994).
// Calculates Pearson correlation between users over items both users have already rated
// Select top k most similar users (neighbors) and predict missing ratings based on neighbors' ratings
class UserBasedCF : public Recommender {
public:
    UserBasedCF(const Dataset& data, int k = 50, int minCoRated = 5);

    double predict(int userId, int movieId) const override;
    std::vector<std::pair<int, double>> recommend(int userId, int k) const override;
    std::string name() const override;

private:
    const Dataset& data;
    int k;  // neighbors kept per movie
    int minCoRated; // ignore pairs w fewer shared movies

    // userId -> average rating (use in similarity and prediction)
    std::unordered_map<int, double> userMean;

    // userId -> top-k (similarUserId, similarity), highest first
    std::unordered_map<int, std::vector<std::pair<int, double>>> neighbors;

    void computeUserMeans();
    void computeSimilarities();
};