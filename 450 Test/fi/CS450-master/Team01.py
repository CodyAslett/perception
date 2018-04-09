"""
Author: Cutler Hollist
Purpose: This program helps the author practice the syntax of Python. It
    defines a Movie class and then performs various operations on it.
"""

from random import Random
import numpy as np

class Movie:
    def __init__(self, title="", year=0, runtime=0):
        self.title = title
        self.year = year
        if runtime > 0:
            self.runtime = runtime
        else:
            self.runtime = 0

    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins"

    def get_runtime(self):
        hours = self.runtime // 60
        minutes = self.runtime % 60
        return hours, minutes


def create_movie_list():
    movies = []

    m_one = Movie("Bob", 1995, 200)
    movies.append(m_one)

    m_two = Movie()
    m_two.title = "Harry"
    m_two.year = 2003
    m_two.runtime = 125
    movies.append(m_two)

    movies.append(Movie("The Last Homely House", 2020, 20))
    movies.append(Movie(year=2000, title="Job", runtime=184))

    return movies


def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, I just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array


def main():
    # Part 1
    movie = Movie()

    movie.title = "Star Wars"
    movie.year = 2017
    movie.runtime = 150

    print(movie)

    hours, minutes = movie.get_runtime()

    print(hours, "h ", minutes, "mins")
    print()

    # Part 2
    movies = create_movie_list()

    print("All Movies:")
    for movie in movies:
        print(movie)
    print()

    long_movies = [movie for movie in movies if movie.runtime > 150]
    print("Long Movies:")
    for movie in long_movies:
        print(movie)
    print()

    print("Movie Ratings:")
    random = Random()
    ratings_map = {}
    for movie in movies:
        rating = random.uniform(0, 5)
        ratings_map[movie.title] = rating
    for title in ratings_map:
        rating = ratings_map[title]
        print("{} - {:.2f} stars".format(title, rating))

    # Part 3
    data = get_movie_data()

    np.set_printoptions(suppress=True)

    print("Data:")
    print(data)
    print("Data Shape")
    print(data.shape)

    rows = data.shape[0]
    print("Rows:", rows)

    cols = data.shape[1]
    print("Cols:", cols)

    print("\nFirst two rows:")
    first_two_rows = data[0:2]
    print(first_two_rows)

    print("\nLast two columns:")
    last_two_cols = data[:, -2:]
    print(last_two_cols)

    print("\nViews:")
    views = data[:, 1]
    print(views)


if __name__ == "__main__":
    main()
