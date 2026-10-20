#pragma once

#include <string>
#include <unordered_map>

// Holds the MovieLens ratings, indexed two ways for fast lookup.
class Dataset {
public:
    using RatingMap = std::unordered_map<int, float>;

    // Loads ratings.dat and movies.dat from folderPath (e.g. "data/ml-1m/").
    // Returns false if the files can't be opened.
    bool load(const std::string& folderPath);

    // Adds a single rating. Used by load(), and by the
    // "rate movies yourself" mode and the train/test split.
    void addRating(int userId, int movieId, float rating);

    // userId -> (movieId -> rating). Used by User-Based CF.
    const std::unordered_map<int, RatingMap>& getUserRatings() const { return userRatings; }

    // movieId -> (userId -> rating). Used by Item-Based CF.
    const std::unordered_map<int, RatingMap>& getMovieRatings() const { return movieRatings; }

    // Movie title, or "Unknown" if the ID isn't found.
    std::string getTitle(int movieId) const;

    size_t numUsers() const { return userRatings.size(); }
    size_t numMovies() const { return movieRatings.size(); }
    size_t numRatings() const { return totalRatings; }

private:
    std::unordered_map<int, RatingMap> userRatings;
    std::unordered_map<int, RatingMap> movieRatings;
    std::unordered_map<int, std::string> movieTitles;
    size_t totalRatings = 0;
};