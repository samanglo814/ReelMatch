// Tests for ItemBasedCF on a small hand-built dataset where the expected
// predictions can be worked out by hand, plus a sanity check on ml-1m.
#include <cmath>
#include <iostream>
#include <string>

#include "Dataset.h"
#include "ItemBasedCF.h"

namespace {
int failures = 0;

void check(bool condition, const std::string& name) {
    std::cout << (condition ? "  PASS  " : "  FAIL  ") << name << "\n";
    if (!condition) failures++;
}

bool near(double a, double b) {
    return std::abs(a - b) < 1e-6;
}

// Users 1-6 rate movies 1, 2 and 4 as x and movie 3 as 6 - x, so movies
// 1, 2, 4 have similarity +1 with each other and -1 with movie 3.
// User 7 rates movies 2 and 4 as 4 and movie 3 as 2, but not movie 1.
void buildToyDataset(Dataset& d) {
    float xs[] = {1, 2, 4, 5, 5, 1};
    for (int u = 0; u < 6; u++) {
        d.addRating(u + 1, 1, xs[u]);
        d.addRating(u + 1, 2, xs[u]);
        d.addRating(u + 1, 4, xs[u]);
        d.addRating(u + 1, 3, 6 - xs[u]);
    }
    d.addRating(7, 2, 4);
    d.addRating(7, 4, 4);
    d.addRating(7, 3, 2);
}
}

int main() {
    std::cout << "Toy dataset tests:\n";
    Dataset toy;
    buildToyDataset(toy);
    ItemBasedCF cf(toy);

    check(cf.name() == "Item-Based CF", "name() is correct");

    // User 7: mean 10/3, deviation 2/3 on movies 2 and 4 -> 10/3 + 2/3 = 4
    check(near(cf.predict(7, 1), 4.0), "predict(7, 1) = 4.0 (worked by hand)");

    // User 1: ratings 1, 1, 1, 5 -> mean 2, deviation -1 on movies 2 and 4 -> 1
    check(near(cf.predict(1, 1), 1.0), "predict(1, 1) = 1.0 (worked by hand)");

    // Unknown user falls back to 3.0
    check(near(cf.predict(99, 1), 3.0), "unknown user returns 3.0");

    // Unknown movie falls back to the user's mean (10/3 for user 7)
    check(near(cf.predict(7, 999), 10.0 / 3.0), "unknown movie returns user's mean");

    // Movie 1 has only 6 ratings (< 20), so recommend() filters it out
    check(cf.recommend(7, 10).empty(), "recommend() skips movies with < 20 ratings");

    std::cout << "\nml-1m tests:\n";
    Dataset real;
    if (!real.load("data/ml-1m/")) {
        std::cout << "  SKIP  data/ml-1m not found\n";
    } else {
        ItemBasedCF realCf(real);
        auto recs = realCf.recommend(1, 10);
        const auto& rated = real.getUserRatings().at(1);

        check(recs.size() == 10, "recommend(1, 10) returns 10 movies");

        bool sorted = true, noneRated = true, inRange = true;
        for (size_t i = 0; i < recs.size(); i++) {
            if (i > 0 && recs[i].second > recs[i - 1].second) sorted = false;
            if (rated.count(recs[i].first)) noneRated = false;
            if (recs[i].second < 1.0 || recs[i].second > 5.0) inRange = false;
        }
        check(sorted, "recommendations sorted highest first");
        check(noneRated, "no already-rated movies recommended");
        check(inRange, "all predicted ratings between 1 and 5");
    }

    std::cout << "\n" << (failures == 0 ? "All tests passed" : "Some tests FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}