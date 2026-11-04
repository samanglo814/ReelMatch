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

