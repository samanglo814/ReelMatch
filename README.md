# ReelMatch

**COP3530 – Project 2**

ReelMatch is a C++ movie recommendation system that uses the MovieLens 1M dataset to generate personalized movie recommendations. It implements and compares two collaborative filtering algorithms: User-Based Collaborative Filtering and Item-Based Collaborative Filtering.

The project evaluates both algorithms based on recommendation accuracy and performance.

**Team (Group 29):** Samantha Anglo (@samanglo814), Josephine Lim (@josephinlim-cmyk), Daniel Su Liu (@danichaa26)

## Features

- Load and process the MovieLens 1M dataset.
- Select a user and view their highly rated movies.
- Generate personalized movie recommendations.
- Compare recommendations from User-Based and Item-Based Collaborative Filtering.
- Evaluate recommendation accuracy using RMSE and Precision@10.
- Benchmark algorithm training times.
- Interactive, menu-driven interface.
- Compare algorithm scalability using datasets of 100K, 250K, 500K, and 1M ratings.

## Recommendation Algorithms

### User-Based Collaborative Filtering

User-Based Collaborative Filtering finds users with similar rating preferences. It uses these similarities to predict how a selected user might rate movies they have not watched.

The implementation uses Pearson correlation to calculate user similarity, with significance weighting and a configurable number of neighbors.

### Item-Based Collaborative Filtering

Item-Based Collaborative Filtering identifies movies that receive similar ratings from users. It uses a user's existing ratings and movie similarities to predict ratings for unseen movies.

The implementation uses adjusted cosine similarity and a configurable number of neighboring movies.

## Dataset

ReelMatch uses the **MovieLens 1M dataset**, which contains:

- 6,040 users
- 3,706 movies
- 1,000,209 ratings

Download the dataset from the [MovieLens website](https://grouplens.org/datasets/movielens/1m/).

After downloading and extracting the dataset, place the required files in the following directory:

```text
ReelMatch/
├── data/
│   └── ml-1m/
│       ├── movies.dat
│       └── ratings.dat
├── include/
├── src/
├── CMakeLists.txt
└── README.md
```

The application expects `movies.dat` and `ratings.dat` inside `data/ml-1m/`. The dataset files are not included in this repository and must be downloaded separately.

## Building and Running

### Requirements

- C++17 compatible compiler
- CMake 3.16 or later
- MovieLens 1M dataset

### Using CLion

1. Clone or download the repository.
2. Open the project in CLion.
3. Make sure the working directory is set to the project root directory.
4. Allow CLion to load and configure the CMake project.
5. Build and run the `ReelMatch` target.

### Using the Command Line

Configure the project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build the application:

```bash
cmake --build build
```

Run the program from the project root folder, so it can find `data/ml-1m/`:

```bash
./build/ReelMatch
```

Running it from inside `build/` will fail to load the dataset.

## Running Tests

After building, run the item-based tests from the project root:

```bash
./build/ItemBasedCFTests
```

These check item-based CF against a small dataset with hand-calculated answers, plus sanity checks on the full ml-1m data.

## Using ReelMatch

After launching the application, use the menu to access the following options:

1. **Select User** – Choose a user from the dataset.
2. **View User's Highly Rated Movies** – Display movies the selected user rated highly.
3. **Generate Recommendations** – Generate personalized movie recommendations.
4. **Compare Recommendation Algorithms** – Compare the top recommendations from both algorithms.
5. **Run Performance Benchmark** – Evaluate RMSE, Precision@10, training time, and dataset scaling for both algorithms.
6. **Exit** – Close the application.

## Evaluation

ReelMatch compares the two recommendation algorithms using the following metrics:

- **RMSE (Root Mean Squared Error):** Measures the difference between predicted ratings and actual ratings. Lower values indicate better rating prediction accuracy.
- **Precision@10:** Measures the proportion of the top 10 recommended movies that are relevant to the user. Higher values indicate better recommendation relevance.
- **Training Time:** Measures how long each algorithm takes to build its recommendation model.

The evaluation uses an 80/20 training and testing split. Ratings are hidden from the training dataset during testing, and movies rated 4 or higher are considered relevant for Precision@10.

## Dataset Scaling Benchmark

To evaluate scalability, ReelMatch compares the training times of User-Based and Item-Based Collaborative Filtering using four dataset sizes: 100K, 250K, 500K, and 1M ratings. Ratings are shuffled using a fixed random seed (42) for reproducibility.

| Ratings | User-Based CF (ms) | Item-Based CF (ms) |
|---|---:|---:|
| 100,000 | 481.98 | 363.39 |
| 250,000 | 2,035.15 | 1,234.04 |
| 500,000 | 10,075.20 | 3,176.27 |
| 1,000,000 | 65,952.95 | 8,598.31 |

Item-Based CF trained faster at every dataset size, about 7.7× faster than User-Based CF at 1M ratings. Results are from a single Release-build run; training times vary by hardware.

## Technologies Used

- C++
- C++ Standard Library
- CMake
- MovieLens 1M Dataset
- Git and GitHub

## Project Structure

```text
ReelMatch/
├── include/
│   ├── Dataset.h
│   ├── Recommender.h
│   ├── UserBasedCF.h
│   ├── ItemBasedCF.h
│   └── Evaluator.h
├── src/
│   ├── main.cpp
│   ├── Dataset.cpp
│   ├── UserBasedCF.cpp
│   ├── ItemBasedCF.cpp
│   └── Evaluator.cpp
├── tests/
│   └── test_item_cf.cpp
├── data/
│   └── ml-1m/
├── CMakeLists.txt
└── README.md
```

## Course

COP3530 – Data Structures and Algorithms  
Project 2
