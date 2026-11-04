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
static const int A = 1, B = 2, C = 3, D = 4;

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
    UserBasedCF cf(d, /*k=*/50, /*minCoRated=*/3);

    // mean(A)=4, mean(B)=3. m4: 4 + (2-3) = 3.   m5: 4 + (4-3) = 5.
    checkNear(cf.predict(A, 4), 3.0, "predictBasic: predict(A, m4)");
    checkNear(cf.predict(A, 5), 5.0, "predictBasic: predict(A, m5)");
}

