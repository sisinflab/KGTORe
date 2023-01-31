from data_preprocessing import movielens_preprocessing2, facebook_book_preprocessing2, yahoo_movies_preprocessing2

movielens_data_folder = './data/movielens'
facebook_book_folder = './data/facebook_book'
yahoo_movies_folder = './data/yahoo_movies'

if __name__ == '__main__':
    #facebook_book_preprocessing2.run(data_folder=facebook_book_folder)
    yahoo_movies_preprocessing2.run(data_folder=yahoo_movies_folder)
    #movielens_preprocessing2.run(data_folder=movielens_data_folder)
