from data_preprocessing import movielens_preprocessing3, facebook_book_preprocessing3, yahoo_movies_preprocessing3

movielens_data_folder = './data/movielens'
facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'

if __name__ == '__main__':
    facebook_book_preprocessing3.run(data_folder=facebook_book_folder)
    #yahoo_movies_preprocessing3.run(data_folder=yahoo_movies_folder)
    #movielens_preprocessing3.run(data_folder=movielens_data_folder)
