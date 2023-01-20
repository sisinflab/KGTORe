from data_preprocessing import movielens_preprocessing, facebook_book_preprocessing, yahoo_movies_preprocessing

movielens_data_folder = './data/movielens'
facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'

if __name__ == '__main__':
    facebook_book_preprocessing.run(data_folder=facebook_book_folder)
    yahoo_movies_preprocessing.run(data_folder=yahoo_movies_folder)
    movielens_preprocessing.run(data_folder=movielens_data_folder)
