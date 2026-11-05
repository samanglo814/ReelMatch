// Tests for UserBasedCF

#include "Dataset.h"
#include "UserBasedCF.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const std::string& label) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << label << "\n";
    }
}

static void checkNear(double actual, double expected, const std::string& label) {
    if (std::abs(actual - expected) > 1e-9) {
        ++failures;
        std::cerr << "FAIL: " << label << " = " << actual << ", expected " << expected << "\n";
    }
}

struct R {
    int user;
    int movie;
    float rating;
};

static Dataset makeDataset(const std::vector<R>& rows) {
    Dataset d;
    for (const auto& r : rows) d.addRating(r.user, r.movie, r.rating);
    return d;
}

// User ids
static const int A = 1, B = 2, C = 3, D = 4, E = 5;

// Main test - A rates m1..m3 = 5,3,4 (mean 4).
// B = A - 1 on those movies (Pearson +1 with A), plus m4 = 2 and m5 = 4, so B's mean is exactly 3
static std::vector<R> mainFixture() {
    return {
            {A, 1, 5}, {A, 2, 3}, {A, 3, 4},
            {B, 1, 4}, {B, 2, 2}, {B, 3, 3}, {B, 4, 2}, {B, 5, 4},
        };
}

static void testPredictBasic() {
    Dataset d = makeDataset(mainFixture());
    UserBasedCF cf(d, 50, 3, 1, 1);

    // mean(A)=4, mean(B)=3. B's rating mapped onto A's scale, then averaged
    // m4: B gave 2 (below its mean): 4 - (3-2) * (4-1)/(3-1) = 2.5
    // m5: B gave 4 (above its mean): 4 + (4-3) * (5-4)/(5-3) = 4.5
    checkNear(cf.predict(A, 4), 2.5, "predictBasic: predict(A, m4)");
    checkNear(cf.predict(A, 5), 4.5, "predictBasic: predict(A, m5)");
}

static void testPredictFallbacks() {
    Dataset d = makeDataset(mainFixture());
    UserBasedCF cf(d, 50, 3, 1, 1);

    checkNear(cf.predict(A, 99), 4.0, "fallbacks: no neighbor rated it -> user's mean");
    checkNear(cf.predict(999, 1), 3.0, "fallbacks: unknown user -> default rating");
}

static void testTopRatingMapsToFive() {
    // B = A - 1 on m1..m3, B mean = 3.5. B gave m4 its top rating (5), which maps to exactly 5.0 on A's scale
    Dataset d = makeDataset({
        {A, 1, 5}, {A, 2, 3}, {A, 3, 4},
        {B, 1, 4}, {B, 2, 2}, {B, 3, 3}, {B, 4, 5},
    });
    UserBasedCF cf(d, 50, 3, 1, 1);
    checkNear(cf.predict(A, 4), 5.0, "clampsHigh: predict(A, m4)");
}

static void testBottomRatingMapsToOne() {
    // A = (2,1,2), B = A + 1, B mean = 2.25. B gave m4 its lowest rating (1), which maps to exactly 1.0 on A's scale.
    Dataset d = makeDataset({
        {A, 1, 2}, {A, 2, 1}, {A, 3, 2},
        {B, 1, 3}, {B, 2, 2}, {B, 3, 3}, {B, 4, 1},
    });
    UserBasedCF cf(d, 50, 3, 1, 1);
    checkNear(cf.predict(A, 4), 1.0, "clampsLow: predict(A, m4)");
}

static void testExcludedNeighbors() {
    // C has zero variance on the shared movies (all 3s) and D = 6 - A (Pearson -1).
    // Neither should become a neighbor, even though both rate m4 = 5.
    Dataset d = makeDataset({
        {A, 1, 5}, {A, 2, 3}, {A, 3, 4},
        {C, 1, 3}, {C, 2, 3}, {C, 3, 3}, {C, 4, 5},
        {D, 1, 1}, {D, 2, 3}, {D, 3, 2}, {D, 4, 5},
    });
    UserBasedCF cf(d, 50, 3, 1, 1);

    checkNear(cf.predict(A, 4), 4.0, "excludedNeighbors: falls back to A's mean");
    check(cf.recommend(A, 5).empty(), "excludedNeighbors: recommend(A) is empty");
}

static void testMinCoRated() {
    // A and B share only m1 and m2.
    std::vector<R> rows = {
        {A, 1, 5}, {A, 2, 3}, {A, 3, 4},
        {B, 1, 4}, {B, 2, 2}, {B, 4, 2},
    };
    Dataset d = makeDataset(rows);

    UserBasedCF strict(d, 50, 3, 1, 1);
    checkNear(strict.predict(A, 4), 4.0, "minCoRated: pair dropped at threshold 3 -> mean(A)");

    UserBasedCF loose(d, 50, 2, 1, 1);
    // Pearson over 2 shared movies is +1. mean(B) = 8/3, B gave 2 (below mean)
    // so 4 - (8/3 - 2) * (4-1)/(8/3 - 1) = 4 - 1.2 = 2.8
    checkNear(loose.predict(A, 4), 2.8, "minCoRated: pair kept at threshold 2");
}

static void testNeighborCountLimit() {
    // k = 1 keeps only the most similar neighbor.
    // B = A - 1 (r = +1). User E is positively correlated with A but less perfectly (r ~ 0.87).
    Dataset d = makeDataset({
        {A, 1, 5}, {A, 2, 3}, {A, 3, 4},
        {B, 1, 4}, {B, 2, 2}, {B, 3, 3}, {B, 4, 2}, {B, 5, 4},
        {E, 1, 5}, {E, 2, 4}, {E, 3, 5}, {E, 4, 1}, {E, 5, 1},
    });
    UserBasedCF cf(d, 1, 3, 1, 1);

    // Only B should count, so the answer matches the single-neighbor fixture.
    checkNear(cf.predict(A, 4), 2.5, "neighborCountLimit: predict(A, m4)");
    checkNear(cf.predict(A, 5), 4.5, "neighborCountLimit: predict(A, m5)");
}

static void testRecommend() {
    Dataset d = makeDataset(mainFixture());
    UserBasedCF cf(d, 50, 3, 1, 1);

    auto recs = cf.recommend(A, 10);
    check(recs.size() == 2, "recommend: only m4 and m5 are unseen and rated by a neighbor");
    if (recs.size() == 2) {
        check(recs[0].first == 5, "recommend: first result is m5 (sorted descending)");
        checkNear(recs[0].second, 4.5, "recommend: score of m5");
        check(recs[1].first == 4, "recommend: second result is m4");
        checkNear(recs[1].second, 2.5, "recommend: score of m4");
    }

    for (const auto& rec : recs) {
        int movieId = rec.first;
        double score = rec.second;
        check(movieId != 1 && movieId != 2 && movieId != 3,
              "recommend: never recommends an already-rated movie");
        checkNear(score, cf.predict(A, movieId),
                  "recommend: score agrees with predict() for m" + std::to_string(movieId));
    }

    auto top1 = cf.recommend(A, 1);
    check(top1.size() == 1, "recommend: k=1 returns one result");
    if (!top1.empty()) check(top1[0].first == 5, "recommend: k=1 returns m5");

    check(cf.recommend(999, 5).empty(), "recommend: unknown user returns empty list");
}

// A: m1..m3 = 4,2,3 (mean 3). B = A - 1 and E = A + 1 on those, so both are Pearson +1 neighbors.
// m4 is rated only by B (5). m5 is rated by both B (2) and E (4).
// mean(B) = (3+1+2+5+2)/5 = 2.6, mean(E) = (5+3+4+4)/4 = 4.0.
static std::vector<R> thinEvidenceFixture() {
    return {
        {A, 1, 4}, {A, 2, 2}, {A, 3, 3},
        {B, 1, 3}, {B, 2, 1}, {B, 3, 2}, {B, 4, 5}, {B, 5, 2},
        {E, 1, 5}, {E, 2, 3}, {E, 3, 4}, {E, 5, 4},
    };
}

static void testMinNeighborsUsed() {
    Dataset d = makeDataset(thinEvidenceFixture());

    // Threshold 1: m4 is backed by B alone. B gave it a 5, which maps to 5.0 on A's scale.
    UserBasedCF loose(d, 50, 3, 1, 1);
    checkNear(loose.predict(A, 4), 5.0, "minNeighborsUsed=1: single-neighbor movie 5-star movie -> 5.0");

    // Threshold 2: m4 falls back to A's mean; m5 (two neighbors) is predicted normally
    // B gave m5 a 2 (mean 2.6): 3 - 0.6 * (3-1)/(2.6-1) = 2.25.  E gave 4 (= E's mean 4.0): 3.0.
    // Equal similarities, so the average is 2.625.
    UserBasedCF strict(d, 50, 3, 1, 2);
    checkNear(strict.predict(A, 4), 3.0, "minNeighborsUsed=2: single-neighbor movie -> user's mean");
    checkNear(strict.predict(A, 5), 2.625, "minNeighborsUsed=2: two-neighbor movie predicted normally");

    // recommend() applies the same rule, so m4 is not recommended at all.
    auto recs = strict.recommend(A, 10);
    check(recs.size() == 1, "minNeighborsUsed=2: recommend returns only the well-supported movie");
    if (!recs.empty()) {
        check(recs[0].first == 5, "minNeighborsUsed=2: recommended movie is m5");
        checkNear(recs[0].second, 2.625, "minNeighborsUsed=2: recommended score matches predict()");
    }
}

static void testMinMovieRatings() {
    Dataset d = makeDataset(thinEvidenceFixture());

    // m4 has 1 rating overall, m5 has 2. With a 2-rating floor, only m5 can be recommended
    UserBasedCF cf(d, 50, 3, 2, 1);
    auto recs = cf.recommend(A, 10);
    check(recs.size() == 1, "minMovieRatings=2: obscure movie is skipped");
    if (!recs.empty()) check(recs[0].first == 5, "minMovieRatings=2: m5 is the only recommendation");

    // predict() is not affected by the movie-popularity floor
    checkNear(cf.predict(A, 4), 5.0, "minMovieRatings=2: predict() unaffected");
}

// situation: a generous user (mean 4.5) with a harsh neighbor (mean 2.5) who loved a movie
// testing mapToUserScale() so that predicted rankings are not all 5.00/5
static void testGenerousUserNotSaturated() {
    // A: mean 4.5 (5,4,5,4). B = A - 2 on the shared movies, plus m4 = 4 (mean(B) = (3+2+3+2+4)/5 = 2.8).
    Dataset d = makeDataset({
        {A, 1, 5}, {A, 2, 4}, {A, 3, 5}, {A, 5, 4},
        {B, 1, 3}, {B, 2, 2}, {B, 3, 3}, {B, 5, 2}, {B, 4, 4},
    });
    UserBasedCF cf(d, 50, 3, 1, 1);

    double p = cf.predict(A, 4);
    // B's 4 is above B's mean 2.8: 4.5 + (4 - 2.8) * (5 - 4.5)/(5 - 2.8) = 4.5 + 1.2 * 0.5/2.2
    checkNear(p, 4.5 + 1.2 * 0.5 / 2.2, "generousUser: predict(A, m4) stays below 5");
    check(p < 5.0, "generousUser: not saturated at 5.0");

    // Bounded for every possible neighbor rating 1..5. (predict() also clamps so this is only a sanity check)
    for (int rating = 1; rating <= 5; ++rating) {
        Dataset e = makeDataset({
            {A, 1, 5}, {A, 2, 4}, {A, 3, 5}, {A, 5, 4},
            {B, 1, 3}, {B, 2, 2}, {B, 3, 3}, {B, 5, 2}, {B, 4, static_cast<float>(rating)},
        });
        UserBasedCF cfe(e, 50, 3, 1, 1);
        double q = cfe.predict(A, 4);
        check(q >= 1.0 && q <= 5.0, "generousUser: prediction within [1, 5] for neighbor rating " + std::to_string(rating));
    }
}

// make sure rec list is sorted in descending order
// and is ordered by the same number it report- same from predict()
static void testRecommendSortedByDisplayedScore() {
    // B and E are both Pearson +1 neighbors of A, and both rate m4..m8 with different values.
    std::vector<R> rows = {
        {A, 1, 4}, {A, 2, 2}, {A, 3, 3},
        {B, 1, 3}, {B, 2, 1}, {B, 3, 2}, {B, 4, 5}, {B, 5, 1}, {B, 6, 3}, {B, 7, 4}, {B, 8, 2},
        {E, 1, 5}, {E, 2, 3}, {E, 3, 4}, {E, 4, 4}, {E, 5, 2}, {E, 6, 5}, {E, 7, 3}, {E, 8, 4},
    };
    Dataset d = makeDataset(rows);
    UserBasedCF cf(d, 50, 3, 2, 2);

    auto recs = cf.recommend(A, 10);
    check(recs.size() == 5, "sorted: all five unseen two-neighbor movies are returned");
    for (size_t i = 0; i < recs.size(); ++i) {
        checkNear(recs[i].second, cf.predict(A, recs[i].first),
                  "sorted: reported score equals predict() for m" + std::to_string(recs[i].first));
        if (i > 0) check(recs[i - 1].second >= recs[i].second, "sorted: scores are non-increasing");
    }
}

int main() {
    testPredictBasic();
    testPredictFallbacks();
    testTopRatingMapsToFive();
    testBottomRatingMapsToOne();
    testExcludedNeighbors();
    testMinCoRated();
    testRecommend();
    testNeighborCountLimit();
    testMinNeighborsUsed();
    testMinMovieRatings();
    testGenerousUserNotSaturated();
    testRecommendSortedByDisplayedScore();

    if (failures == 0) {
        std::cout << "All UserBasedCF tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
