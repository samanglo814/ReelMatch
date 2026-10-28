#include "ItemBasedCF.h"

ItemBasedCF::ItemBasedCF(const Dataset& data, int k) : data(data), k(k) {
    computeUserMeans();
    computeSimilarities();
}

std::string ItemBasedCF::name() const {
    return "Item-Based CF";
}

// Computes each user's average rating. Adjusted cosine subtracts this
// from every rating so harsh and generous raters are compared fairly.
// Time: O(R), where R is the total number of ratings.
void ItemBasedCF::computeUserMeans() {
    for (const auto& [userId, ratings] : data.getUserRatings()) {
        if (ratings.empty()) continue;

        double sum = 0.0;
        for (const auto& [movieId, rating] : ratings) {
            sum += rating;
        }
        userMean[userId] = sum / static_cast<double>(ratings.size());
    }
}

void ItemBasedCF::computeSimilarities() {
    // TODO
}

double ItemBasedCF::predict(int userId, int movieId) const {
    (void)userId; (void)movieId;  // silences unused warnings until implemented
    return 0.0;
}

std::vector<std::pair<int, double>> ItemBasedCF::recommend(int userId, int k) const {
    (void)userId; (void)k;
    return {};
}