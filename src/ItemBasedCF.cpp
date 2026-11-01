#include "ItemBasedCF.h"

#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>

namespace {
    struct PairSums {
        double dot = 0.0;  // Sum of centered(i) * centered(j)
        double normI = 0.0;  // Sum of centered(i)^2 
        double normJ = 0.0;  // Sum of centered(j)^2 
        int count = 0;  // Number of users who rated both i and j
    };
    long long pairKey(int i, int j) {
        return static_cast<long long>(i) * 100000LL + j;
    }
}

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

// Adjusted cosine similarity (Sarwar et al.) between every pair of movies
// rated by at least one common user, keeping the top k per movie.
// Time: O(sum over users of n_u^2 + M * m log m), where n_u = ratings by
// user u, M = number of movies, m = neighbors per movie before trimming.
void ItemBasedCF::computeSimilarities() {
    std::unordered_map<long long, PairSums> sums;

    for (const auto& [userId, ratings] : data.getUserRatings()) {
        double mean = userMean.at(userId);

        // Copy into a vector so we can visit each pair once by index.
        std::vector<std::pair<int, float>> rated(ratings.begin(), ratings.end());

        for (size_t a = 0; a < rated.size(); a++) {
            for (size_t b = a + 1; b < rated.size(); b++) {
                int i = rated[a].first;
                int j = rated[b].first;
                double ci = rated[a].second - mean;  // centered ratings
                double cj = rated[b].second - mean;
                if (i > j) { std::swap(i, j); std::swap(ci, cj); }

                PairSums& s = sums[pairKey(i, j)];
                s.dot += ci * cj;
                s.normI += ci * ci;
                s.normJ += cj * cj;
                s.count++;
            }
        }
    }

    // Convert each pair's sums into a similarity and record it for both movies.
    for (const auto& [key, s] : sums) {
        if (s.count < 5) continue; // Ignore pairs with too few common raters
        double denom = std::sqrt(s.normI) * std::sqrt(s.normJ);
        if (denom == 0.0) continue;  // no variation, similarity undefined

        int i = static_cast<int>(key / 100000LL);
        int j = static_cast<int>(key % 100000LL);
        double sim = s.dot / denom;

        neighbors[i].push_back({j, sim});
        neighbors[j].push_back({i, sim});
    }

    // Keep only the k most similar movies for each movie, highest first.
    // Storing every pair would use far more memory and slow down predict().
    for (auto& [movieId, list] : neighbors) {
        std::sort(list.begin(), list.end(),
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });
        if (list.size() > static_cast<size_t>(k)) {
            list.resize(k);
        }
    }
}

double ItemBasedCF::predict(int userId, int movieId) const {
    (void)userId; (void)movieId;  // silences unused warnings until implemented
    return 0.0;
}

std::vector<std::pair<int, double>> ItemBasedCF::recommend(int userId, int k) const {
    (void)userId; (void)k;
    return {};
}