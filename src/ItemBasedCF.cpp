#include "ItemBasedCF.h"

ItemBasedCF::ItemBasedCF(const Dataset& data, int k) : data(data), k(k) {
    computeUserMeans();
    computeSimilarities();
}

std::string ItemBasedCF::name() const {
    return "Item-Based CF";
}

void ItemBasedCF::computeUserMeans() {
    // TODO
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