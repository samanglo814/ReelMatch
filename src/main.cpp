
#include <iostream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <memory>

#include "Evaluator.h"
#include "Dataset.h"
#include "UserBasedCF.h"
#include "ItemBasedCF.h"

using namespace std;

int main() {

    // Load MovieLens dataset
    Dataset dataset;

    cout << "Loading MovieLens dataset..." << endl;

    if (!dataset.load("data/ml-1m/")) {
        cout << "Error: Could not load dataset files." << endl;
        return 1;
    }

    cout << "Dataset loaded successfully!" << endl;
    cout << "Users: " << dataset.numUsers() << endl;
    cout << "Movies: " << dataset.numMovies() << endl;
    cout << "Ratings: " << dataset.numRatings() << endl;

    // Store the currently selected user, -1 represents that nobody is selected yet
    int selectedUserId = -1;

    int choice = 0;

    // Store the recommendation models so they only need to be built once
    unique_ptr<UserBasedCF> userBased;
    unique_ptr<ItemBasedCF> itemBased;

    // Main menu loop
    while (choice != 6) {

        cout << "\n==============================================\n";
        cout << "                  REELMATCH\n";
        cout << "          Movie Recommendation System\n";
        cout << "==============================================\n";
        cout << "1. Select User\n";
        cout << "2. View User's Highly Rated Movies\n";
        cout << "3. Generate Recommendations\n";
        cout << "4. Compare Recommendation Algorithms\n";
        cout << "5. Run Performance Benchmark\n";
        cout << "6. Exit\n";
        cout << "Enter selection: ";

        cin >> choice;

        // Check for invalid input
        if (cin.fail()) {
            cout << "Invalid input. Please enter a number.\n";

            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');

            continue;
        }

        switch (choice) {

            // 1. Select User
            case 1: {
                int userId;

                cout << "\nEnter User ID (1-" << dataset.numUsers() << "): ";
                cin >> userId;

                if (cin.fail()) {
                    cout << "Invalid input. Please enter a number.\n";

                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');

                    break;
                }

                // Check whether the user exists
                auto it = dataset.getUserRatings().find(userId);

                if (it == dataset.getUserRatings().end()) {
                    cout << "User not found. Please try again.\n";
                }
                else {
                    selectedUserId = userId;

                    cout << "User " << selectedUserId
                         << " selected successfully!\n";
                }

                break;
            }


            // 2. View User's Highly Rated Movies
            case 2: {
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                }
                else {
                    // Get all ratings from the selected user
                    const auto& userRatings = dataset.getUserRatings().at(selectedUserId);

                    // Store movies rated 4.0 or higher
                    vector<pair<int, float>> highlyRatedMovies;

                    for (const auto& rating : userRatings) {
                        if (rating.second >= 4.0) {
                            highlyRatedMovies.push_back(rating);
                        }
                    }

                    // Sort movies by rating, highest first
                    sort(highlyRatedMovies.begin(), highlyRatedMovies.end(),
                         [](const pair<int, float>& a, const pair<int, float>& b) {
                             return a.second > b.second;
                         });

                    cout << "\nHighly Rated Movies for User "
                         << selectedUserId << ":\n";

                    if (highlyRatedMovies.empty()) {
                        cout << "No highly rated movies found.\n";
                    }
                    else {
                        for (const auto& movie : highlyRatedMovies) {
                            int movieId = movie.first;
                            float rating = movie.second;

                            cout << fixed << setprecision(1);
                            cout << dataset.getTitle(movieId)
                                 << " - Rating: " << rating << "/5.0\n";
                        }

                        cout << "\nTotal highly rated movies: "
                             << highlyRatedMovies.size() << endl;
                    }
                }
                break;
            }


            // 3. Generate Recommendations
            case 3: {
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                    break;
                }

                int algorithmChoice;

                cout << "\nSelect Recommendation Algorithm:\n";
                cout << "1. User-Based Collaborative Filtering\n";
                cout << "2. Item-Based Collaborative Filtering\n";
                cout << "Enter selection: ";

                cin >> algorithmChoice;

                if (cin.fail()) {
                    cout << "Invalid input. Please enter a number.\n";
                    cin.clear();
                    cin.ignore(numeric_limits<streamsize>::max(), '\n');
                    break;
                }

                vector<pair<int, double>> recommendations;

                if (algorithmChoice == 1) {
                    // Build User-Based CF model if it has not been built yet
                    if (!userBased) {
                        cout << "\nBuilding User-Based CF model. Please wait...\n";
                        userBased = make_unique<UserBasedCF>(dataset);
                        cout << "User-Based CF model ready!\n";
                    }

                    recommendations = userBased->recommend(selectedUserId, 10);
                }
                else if (algorithmChoice == 2) {
                    // Build Item-Based CF model if it has not been built yet
                    if (!itemBased) {
                        cout << "\nBuilding Item-Based CF model. Please wait...\n";
                        itemBased = make_unique<ItemBasedCF>(dataset);
                        cout << "Item-Based CF model ready!\n";
                    }

                    recommendations = itemBased->recommend(selectedUserId, 10);
                }
                else {
                    cout << "Invalid algorithm selection.\n";
                    break;
                }

                cout << "\nTop 10 Movie Recommendations for User "
                     << selectedUserId << ":\n";

                if (recommendations.empty()) {
                    cout << "No recommendations available.\n";
                }
                else {
                    cout << fixed << setprecision(2);

                    for (const auto& movie : recommendations) {
                        cout << dataset.getTitle(movie.first)
                             << " - Predicted Rating: "
                             << movie.second << "/5.0\n";
                    }
                }

                break;
            }


            // 4. Compare Recommendation Algorithms
            case 4: {
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                    break;
                }

                // Build User-Based CF model if it has not been built yet
                if (!userBased) {
                    cout << "\nBuilding User-Based CF model. Please wait...\n";
                    userBased = make_unique<UserBasedCF>(dataset);
                    cout << "User-Based CF model ready!\n";
                }

                // Build Item-Based CF model if it has not been built yet
                if (!itemBased) {
                    cout << "\nBuilding Item-Based CF model. Please wait...\n";
                    itemBased = make_unique<ItemBasedCF>(dataset);
                    cout << "Item-Based CF model ready!\n";
                }

                // Generate recommendations from both algorithms
                vector<pair<int, double>> userRecommendations =
                    userBased->recommend(selectedUserId, 10);

                vector<pair<int, double>> itemRecommendations =
                    itemBased->recommend(selectedUserId, 10);

                // Display User-Based CF recommendations
                cout << "\n==============================================\n";
                cout << "       USER-BASED CF RECOMMENDATIONS\n";
                cout << "==============================================\n";

                if (userRecommendations.empty()) {
                    cout << "No recommendations available.\n";
                }
                else {
                    cout << fixed << setprecision(2);

                    for (size_t i = 0; i < userRecommendations.size(); i++) {
                        cout << i + 1 << ". "
                             << dataset.getTitle(userRecommendations[i].first)
                             << " - Predicted Rating: "
                             << userRecommendations[i].second << "/5.0\n";
                    }
                }

                // Display Item-Based CF recommendations
                cout << "\n==============================================\n";
                cout << "       ITEM-BASED CF RECOMMENDATIONS\n";
                cout << "==============================================\n";

                if (itemRecommendations.empty()) {
                    cout << "No recommendations available.\n";
                }
                else {
                    cout << fixed << setprecision(2);

                    for (size_t i = 0; i < itemRecommendations.size(); i++) {
                        cout << i + 1 << ". "
                             << dataset.getTitle(itemRecommendations[i].first)
                             << " - Predicted Rating: "
                             << itemRecommendations[i].second << "/5.0\n";
                    }
                }

                break;
            }


            // 5. Run Performance Benchmark
            case 5: {
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                }
                else {
                    Evaluator evaluator(dataset);

                    // Evaluate recommendation accuracy and training time.
                    evaluator.run(selectedUserId);

                    // Compare training times using different dataset sizes.
                    evaluator.runScalingBenchmark();
                }
                break;
            }


            // 6. Exit
            case 6:
                cout << "\nThank you for using ReelMatch!\n";
                break;

            default:
                cout << "Invalid selection. Please choose 1-6.\n";
                break;
        }
    }

    return 0;
}
