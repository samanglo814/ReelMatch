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
    const double kEps = 1e-9; // help with floating point comparisons
    const double kRankShrinkage = 1.0; // recommend() ranking: larger = favor movies with more neighbor support

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

UserBasedCF::UserBasedCF(const Dataset& data, int k, int minCoRated, int minMovieRated, int minNeighborsUsed) : data(data), k(k), minCoRated(minCoRated), minMoviesRated(minMoviesRated), minNeighborsUsed(minNeighborsUsed) {
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

// time complexity: O(Σₘ nₘ²), where nₘ is number of users who have rated movie m
void UserBasedCF::computeSimilarities() {
    // use pearson's coefficient to calculate
    const auto& userRatings = data.getUserRatings();
    const auto& movieRatings = data.getMovieRatings();

    for (const auto& [u, uMovies] : userRatings)
    {
        // go through u's movies and for every movie walk through everyone else who watched it
        // only users sharing at least one movie with u get an Accumulator to keep track of their similarity
        std::unordered_map<int, Accumulator> acc;
        for (const auto& [movieId, ru] : uMovies)
        {
            auto it = movieRatings.find(movieId);
            if (it == movieRatings.end()) continue;
            for (const auto& [v, rv] : it->second)
            {
                if (v == u) continue; // don't compare user to itself
                acc[v].add(ru, rv);
            }
        }

        // use sums calculated by Accumulator and plug into Pearson correlation formula
        std::vector<std::pair<int, double>> candidates;
        candidates.reserve(acc.size()); // preallocate memory without changing vector size or initializing elements
        for (const auto& [v, a] : acc)
        {
            if (a.n < minCoRated) continue;

            const double n = static_cast<double>(a.n);
            const double num = a.sxy - a.sx * a.sy / n; // covariance numerator
            const double dx = a.sxx - a.sx * a.sx / n;
            const double dy = a.syy - a.sy * a.sy / n;
            if (dx <= kEps || dy <= kEps) continue; // skip pair if essentially zero variance

            double sim = num / std::sqrt(dx * dy);

            // significance weighting: ignore pairs with few shared movies
            sim *= std::min(a.n, kSignificanceCap) / static_cast<double>(kSignificanceCap);

            //keep only positively correlated neighbors
            if (sim > 0.0) candidates.emplace_back(v, sim);

        }

        // partial sort for top k, highest similarity first
        const size_t keep = std::min(static_cast<size_t>(std::max(k, 0)), candidates.size());
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
            [](const auto &a, const auto& b)
            {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first; // use userId as tiebreak
            });
        candidates.resize(keep);
        neighbors[u] = std::move(candidates);
    }



}

double UserBasedCF::predict(int userId, int movieId) const {
    auto meanIt = userMean.find(userId);
    const double base = (meanIt != userMean.end()) ? meanIt->second : kDefaultRating; // fallback if user has no data

    auto nbIt = neighbors.find(userId);
    if (nbIt == neighbors.end()) return base;

    const auto& userRatings = data.getUserRatings();

    double num = 0, den = 0;
    int used = 0; // how many neighbors actually rated this movie
    for (const auto& [v, sim] : nbIt->second)
    {
        auto vIt = userRatings.find(v);
        if (vIt == userRatings.end()) continue;
        auto rIt = vIt->second.find(movieId);
        if (rIt == vIt->second.end()) continue; // neighbor didn't rate this movie

        num += sim * (rIt->second - userMean.at(v));
        den += std::abs(sim);
        ++used;
    }

    if (used < minNeighborsUsed || den < kEps) return base; // no neighbor rated it or too little evidence (only one neighbor): fall back to user mean
    return std::clamp(base + num / den, kMinRating, kMaxRating);
}

std::vector<std::pair<int, double>> UserBasedCF::recommend(int userId, int k) const {
    auto meanIt = userMean.find(userId);
    auto nbIt = neighbors.find(userId);
    if (meanIt == userMean.end() || nbIt == neighbors.end()) return {};

    const auto& userRatings = data.getUserRatings();
    const auto& seen = userRatings.at(userId);

    // do one pass over neighbors, accumulating num and denom of prediction formula for every movie user hasn't rated
    struct Score {double num = 0, den = 0; int used = 0;};
    std::unordered_map<int, Score> scores;
    const auto& movieRatings = data.getMovieRatings();

    for (const auto& [v, sim] : nbIt->second)
    {
        auto vIt = userRatings.find(v);
        if (vIt == userRatings.end()) continue;
        const double vMean = userMean.at(v);
        for (const auto& [movieId, r] : vIt->second)
        {
            if (seen.count(movieId)) continue;

            //skip obscure movies with too few ratings
            auto mIt = movieRatings.find(movieId);
            if (mIt == movieRatings.end() || static_cast<int>(mIt->second.size()) < minMoviesRated) continue;

            Score& s = scores[movieId];
            s.num += sim * (r - vMean);
            s.den += std::abs(sim);
            ++s.used;
        }
    }

    const double base = meanIt->second;

    // rank by shrunk, unclamped score so movies ranked by many neighbors beat movies backed by single neighbor
    // clamping no longer creates ties full of 5.0
    // score reported is still the same clamped value that predict() returns though
    struct Candidate { int movieId; double rankScore; double predicted;};
    std::vector<Candidate> candidates;
    candidates.reserve(scores.size());
    for (const auto& [movieId, s] : scores)
    {
        if (s.used < minNeighborsUsed || s.den < kEps) continue;
        candidates.push_back({movieId, base + s.num / (s.den + kRankShrinkage), std::clamp(base + s.num / s.den, kMinRating, kMaxRating)});
    }

    const size_t keep = std::min(static_cast<size_t>(std::max(k, 0)), candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                      [](const auto& a, const auto& b) {
                          if (a.rankScore != b.rankScore) return a.rankScore > b.rankScore;
                          return a.movieId < b.movieId;
                      });

    std::vector<std::pair<int,double>> results;
    results.reserve(keep);
    for (size_t i = 0; i < keep; ++i) results.emplace_back(candidates[i].movieId, candidates[i].predicted);
    return results;
}