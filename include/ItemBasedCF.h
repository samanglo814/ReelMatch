#pragma once

#include "Dataset.h"
#include "Recommender.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Item-Based Collaborative Filtering (Sarwar et al., 2001).
// Precomputes the top-k most similar movies for every movie using
// adjusted cosine similarity, then predicts a user's rating for a movie
// from their ratings of its most similar movies.
class ItemBasedCF : public Recommender {
public:
    ItemBasedCF(const Dataset& data, int k = 50);

    double predict(int userId, int movieId) const override;
    std::vector<std::pair<int, double>> recommend(int userId, int k) const override;
    std::string name() const override;

private:
    const Dataset& data;
    int k;  // neighbors kept per movie

    // userId -> average rating (adjusted cosine subtracts this)
    std::unordered_map<int, double> userMean;

    // movieId -> top-k (similarMovieId, similarity), highest first
    std::unordered_map<int, std::vector<std::pair<int, double>>> neighbors;

    void computeUserMeans();
    void computeSimilarities();
};