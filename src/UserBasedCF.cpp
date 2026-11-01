#include "UserBasedCF.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace
{
    const double kDefaultRating = 3.0; // in case user has no data
    const double kMinRating = 1.0;
    const double kMaxRating  = 5.0;
    const int kSignificanceCap = 50; // pair with 50+ shared movies get factor of 1.0 (max pearson coefficient)
    const int kEps = 1e-9; // help with floating point comparisons

    // run sums for a pair of movies (u,v) to make it easier to computer Pearson later
    // x = u's rating, y = v's rating
    struct Accumulator
    {
        int n = 0; // count
        double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0; //sx, sy = sums of each user's ratings; sxy, sxx, syy = sums of products and squares

        void add(double x, double y)
        {
            ++n;
            sx += x;
            sy += y;
            sxy += x * y;
            sxx += x * x;
            syy += y * y;
        }
    };
}
UserBasedCF::UserBasedCF(const Dataset& data, int k, int minCoRated) : data(data), k(k), minCoRated(minCoRated) {
    computeUserMeans();
    computeSimilarities();
}

std::string UserBasedCF::name() const {
    return "User-Based CF";
}

// Computes each user's average rating. Pearson coefficient subtracts this
// from every rating so harsh and generous raters are compared fairly.
// Time: O(R), where R is the total number of ratings.
void UserBasedCF::computeUserMeans() {
    for (const auto& [userId, ratings] : data.getUserRatings()) {
        if (ratings.empty()) continue;

        double sum = 0.0;
        for (const auto& [movieId, rating] : ratings) {
            sum += rating;
        }
        userMean[userId] = sum / static_cast<double>(ratings.size());
    }
}

void UserBasedCF::computeSimilarities() {
    // use pearson's coefficient to calculate

}

double UserBasedCF::predict(int userId, int movieId) const {
    (void)userId; (void)movieId;  // silences unused warnings until implemented
    return 0.0;
}

std::vector<std::pair<int, double>> UserBasedCF::recommend(int userId, int k) const {
    (void)userId; (void)k;
    return {};
}