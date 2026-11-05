#include "Dataset.h"

#include <fstream>
#include <string>
#include <exception>

using namespace std;

// File ratings.dat
// 2::1357::5::978298709
// UserID::MovieID::Rating::Timestamp

// File movies.dat
// 1::Toy Story (1995)::Animation|Children's|Comedy
// MovieID::Title::Genres


// Time Complexity: O(M + R), where M is the number of movies
// and R is the number of ratings in the dataset.
// Loads movie titles and ratings from the MovieLens files.
bool Dataset::load(const std::string& folderPath) {

    string path = folderPath;
    string line;

    // Makes sure the path is correct: no empty folder or extra slash
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path += "/";
    }

    // Open dataset files
    ifstream ratingsFile(path + "ratings.dat");
    ifstream moviesFile(path + "movies.dat");

    // Check if files opened successfully
    if (!ratingsFile.is_open() || !moviesFile.is_open()) {
        return false;
    }

    // Clear previous data
    movieTitles.clear();
    userRatings.clear();
    movieRatings.clear();
    totalRatings = 0;

    // Time Complexity: O(M), where M is the number of movies.
    // Extracts movie IDs and titles from movies.dat.
    while (getline(moviesFile, line)) {

        //Gets MovieID
        size_t firstSeparator = line.find("::");

        // Skip to next if substring is not found
        if (firstSeparator == string::npos) {
            continue;
        }

        //Gets Title
        size_t secondSeparator = line.find("::", firstSeparator + 2);

        if (secondSeparator == string::npos) {
            continue;
        }

        // Extracting MovieID and title
        int movieId = stoi(line.substr(0, firstSeparator));
        string title = line.substr(firstSeparator + 2,secondSeparator - firstSeparator - 2); //skips 2 colons

        //Stores in map
        movieTitles[movieId] = title;
    }


    // Time Complexity: O(N), where N is the number of ratings.
    // Extracts user IDs, movie IDs, and ratings from ratings.dat.
    while (getline(ratingsFile, line)) {

        //UserID
        size_t firstSeparator = line.find("::");

        if (firstSeparator == string::npos) {
            continue;
        }

        //MovieID
        size_t secondSeparator = line.find("::", firstSeparator + 2);

        if (secondSeparator == string::npos) {
            continue;
        }

        //Rating
        size_t thirdSeparator = line.find("::", secondSeparator + 2);

        if (thirdSeparator == string::npos) {
            continue;
        }

        // Extracting UserID, MovieID and Rating
        int userId = stoi(line.substr(0, firstSeparator));
        int movieId = stoi(line.substr(firstSeparator + 2,secondSeparator - firstSeparator - 2));
        float rating = stof(line.substr(secondSeparator + 2,thirdSeparator - secondSeparator - 2));

        addRating(userId, movieId, rating);
    }

    return true;
}


// Average Time Complexity: O(1) per rating.
// Inserts a rating into both the user and movie maps.
void Dataset::addRating(int userId, int movieId, float rating) {
    // Outer map: user ID as its key
    // Inner map: user's movie ratings.
    userRatings[userId][movieId] = rating;

    // Outer map: movie ID as its key
    // Inner map: users who rated it
    movieRatings[movieId][userId] = rating;

    totalRatings++;
}


// Average Time Complexity: O(1) per rating.
// Removes a rating from both the user and movie maps.
void Dataset::removeRating(int userId, int movieId) {
    // Find the user's ratings
    auto userIt = userRatings.find(userId);

    if (userIt == userRatings.end()) {
        return;
    }

    // Find the movie rating
    auto ratingIt = userIt->second.find(movieId);

    if (ratingIt == userIt->second.end()) {
        return;
    }

    // Remove the rating from the user's map
    userIt->second.erase(ratingIt);

    // Remove the rating from the movie's map
    auto movieIt = movieRatings.find(movieId);

    if (movieIt != movieRatings.end()) {
        movieIt->second.erase(userId);

        // Remove the movie if nobody has rated it
        if (movieIt->second.empty()) {
            movieRatings.erase(movieIt);
        }
    }

    // Remove the user if they have no remaining ratings
    if (userIt->second.empty()) {
        userRatings.erase(userIt);
    }

    totalRatings--;
}


// Average Time Complexity: O(1).
// Searches for and returns a movie title using its ID.
string Dataset::getTitle(int movieId) const {

    // Search for the movie ID
    auto it = movieTitles.find(movieId);

    // Check if the movie exists
    if (it != movieTitles.end()) {
        return it->second; //return movie title value
    }

    // Movie ID not found
    return "Unknown";
}