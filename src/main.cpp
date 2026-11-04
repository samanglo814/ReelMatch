
#include <iostream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "Dataset.h"

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
            case 2:
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

            case 3:
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                }
                else {
                    cout << "Recommendation functionality coming soon.\n";
                }
                break;

            case 4:
                if (selectedUserId == -1) {
                    cout << "Please select a user first.\n";
                }
                else {
                    cout << "Algorithm comparison coming soon.\n";
                }
                break;

            case 5:
                cout << "Performance benchmark coming soon.\n";
                break;

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
