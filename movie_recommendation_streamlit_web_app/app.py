
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import re
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

item_similarities = pickle.load(open('item_similarities.pkl', 'rb'))

merge_1_sorted_popularity_dict = pickle.load(open('merge_1_sorted_popularity_dict.pkl', 'rb'))

merge_1_sorted_popularity= pd.DataFrame(merge_1_sorted_popularity_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

content_df_genre_dict = pickle.load(open('content_df_genre_dict.pkl','rb'))
content_df_genre= pd.DataFrame(content_df_genre_dict)

csr_data = pickle.load(open('csr_data.pkl','rb'))
knn.fit(csr_data)

ratings_small_collab_pivot = pickle.load(open('ratings_small_collab_pivot.pkl', 'rb'))

content_df_genre_dict = pickle.load(open('content_df_genre_dict.pkl', 'rb'))
content_df_genre = pd.DataFrame(content_df_genre_dict)

st.title('Movie Recommender System')

# Sidebar navigation
page = st.sidebar.selectbox('Select a Recommender', ['Popularity-Based', 'Content-Based', 'Item-Item Based', 'User-User Based','Genre-Based','Hybrid-Model'])

if page == 'Popularity-Based':
    st.subheader('Popularity-Based Recommender')
    selected_movie_name_popularity = st.selectbox('Select a movie', movies['title'].values)

    def fetch_poster(movie_id):
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3cd07b4b483e1b50f7554420fda72adb'.format(movie_id))
        data = response.json()
        return "https://image.tmdb.org/t/p/original" + data['poster_path']

    def extract_nearest_popular_movies(movie_name):
        # Filtering movies that contain the given movie name in their titles
        filtered_movies = merge_1_sorted_popularity[merge_1_sorted_popularity['title'].str.contains(movie_name, case=False)]

        if filtered_movies.empty:
            return "No movies found with that name. Please try a different movie."

        # Get the popularity score of the given movie
        popularity_score = filtered_movies.iloc[0]['popularity']

        # Calculate the absolute difference between the popularity scores of all movies and the given movie
        merge_1_sorted_popularity['popularity_difference'] = abs(merge_1_sorted_popularity['popularity'] - popularity_score)

        # Sort movies based on the popularity difference in ascending order
        sorted_movies = merge_1_sorted_popularity.sort_values('popularity_difference')

        # Exclude the exact movie from the recommendations
        sorted_movies = sorted_movies[sorted_movies['title'] != movie_name]

        # Get the top 5 nearest popular movies
        recommended_movies = sorted_movies.head(5)

        return recommended_movies['title']

    if st.button('Recommend'):
        names = extract_nearest_popular_movies(selected_movie_name_popularity)
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, name in enumerate(names):
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4 if i == 3 else col5:
                st.text(name)
                movie_id = movies[movies['title'] == name]['id'].values[0]
                poster_url = fetch_poster(movie_id)
                st.image(poster_url)

elif page == 'Content-Based':
    st.subheader('Content-Based Recommender')
    selected_movie_name_content = st.selectbox('Select a movie', movies['title'].values)

    def fetch_poster(movie_id):
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3cd07b4b483e1b50f7554420fda72adb'.format(movie_id))
        data = response.json()
        return "https://image.tmdb.org/t/p/original" + data['poster_path']

    def recommend_content(movie):
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_movie_posters = []

        for i in movies_list:
            movie_id = movies.iloc[i[0]].id
            recommended_movies.append(movies.iloc[i[0]].title)

            # Fetch poster from API
            recommended_movie_posters.append(fetch_poster(movie_id))
        return recommended_movies, recommended_movie_posters

    if st.button('Recommend'):
        names, posters = recommend_content(selected_movie_name_content)
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, name in enumerate(names):
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4 if i == 3 else col5:
                st.text(name)
                st.image(posters[i])

elif page == 'Item-Item Based':
    st.subheader('Item-Item Based Recommender')
    selected_movie_name = st.selectbox('Select a movie', movies['title'].values)

    def fetch_poster(movie_id):
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3cd07b4b483e1b50f7554420fda72adb'.format(movie_id))
        data = response.json()
        return "https://image.tmdb.org/t/p/original" + data['poster_path']

    def get_nearest_movies(movie_title):
        movie_id = movies[movies['title'] == movie_title]['id'].values[0]
        similarity_scores = item_similarities[movie_id]
        similar_movie_indices = similarity_scores.argsort()[::-1][1:5]
        similar_movie_data = movies[movies['id'].isin(similar_movie_indices)][['title', 'id']].values
        similar_movies = []
        for title, movie_id in similar_movie_data:
            poster_url = fetch_poster(movie_id)
            similar_movies.append((title, poster_url))
        return similar_movies

    if st.button('Recommend'):
        nearest_movies = get_nearest_movies(selected_movie_name)
        col1, col2, col3, col4 = st.columns(4)
        for i, (title, poster_url) in enumerate(nearest_movies):
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4:
                st.write(title)
                st.image(poster_url, width=200)

elif page == 'User-User Based':
    st.subheader('User-User Based Recommender')
    selected_movie_name = st.selectbox('Select a movie', movies['title'].values)

    def fetch_poster(movie_id):
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=3cd07b4b483e1b50f7554420fda72adb'.format(movie_id))
        data = response.json()
        return "https://image.tmdb.org/t/p/original" + data['poster_path']

    def get_movie_recommendation_user(movie_name):
        n_movies_to_recommend = 10
        pattern = re.compile(f".*{movie_name}.*", re.IGNORECASE)
        movie_list = movies[movies['title'].str.match(pattern)]

        if not movie_list.empty:
            movie_idx = movie_list.iloc[0]['id']

            if movie_idx in movies['id'].values and movie_idx in ratings_small_collab_pivot['id'].values:
                movie_idx = ratings_small_collab_pivot[ratings_small_collab_pivot['id'] == movie_idx].index[0]
                distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
                rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                           key=lambda x: x[1])[:0:-1]

                recommend_frame = []
                for val in rec_movie_indices:
                    movie_idx = ratings_small_collab_pivot.iloc[val[0]]['id']
                    idx = movies[movies['id'] == movie_idx].index.tolist()

                    if idx:
                        recommend_frame.append(
                            {'id': movie_idx, 'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})

                n_recommendations = min(n_movies_to_recommend, len(recommend_frame))

                if recommend_frame:
                    df = pd.DataFrame(recommend_frame[:n_recommendations])
                    return df
                else:
                    return pd.DataFrame()  # Return an empty DataFrame
            else:
                return pd.DataFrame()  # Return an empty DataFrame
        else:
            return pd.DataFrame()  # Return an empty DataFrame

    if st.button('Recommend'):
        nearest_movies_user = get_movie_recommendation_user(selected_movie_name)
        col1, col2, col3, col4 = st.columns(4)
        for i, row in nearest_movies_user.iterrows():
            title = row['Title']
            poster_url = fetch_poster(row['id'])
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4:
                st.write(title)
                st.image(poster_url, width=200)

elif page == 'Genre-Based':
    st.subheader('Genre-Based Recommender')
    selected_genre = st.selectbox('Select a genre', content_df_genre['genres_list'].explode().unique())


    def fetch_poster(movie_id):
        api_key = '3cd07b4b483e1b50f7554420fda72adb'  # Replace with your TMDb API key
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
        data = response.json()
        if 'poster_path' in data:
            return "https://image.tmdb.org/t/p/original" + data['poster_path']
        else:
            return "https://example.com/default_poster.jpg"  # Replace with your default image URL or handle the error in your preferred way


    def get_genres_movies(genre):
        genre_movies = content_df_genre[content_df_genre['genres_list'].apply(lambda x: genre in x)]
        genre_movies = genre_movies.sort_values('score', ascending=False)
        popular_movies = genre_movies.head(5)[['id', 'original_title_x']].values.tolist()

        recommended_genre_movies = []

        for movie_info in popular_movies:
            movie_id = movie_info[0]
            original_title = movie_info[1]
            recommended_genre_movies.append(original_title)

            # Fetch poster from API
            poster_url = fetch_poster(movie_id)
            st.write(original_title)
            st.image(poster_url)

        return recommended_genre_movies

    if st.button('Recommend'):
        recommended_movies_genre = get_genres_movies(selected_genre)
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, movie in enumerate(recommended_movies_genre):
            poster_url = fetch_poster(recommended_movies_genre[i])  # Fetch poster for each movie
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4 if i == 3 else col5:
                st.text(movie)
                st.image(poster_url, width=200)


elif page == 'Hybrid-Model':
    st.subheader('Hybrid Model Recommender')
    selected_movie_name_hybrid = st.selectbox('Select a movie', movies['title'].values, key='hybrid')


    def hybrid_recommender(movie_title):
        # Content-based recommendations
        content_movie_index = movies[movies['title'] == movie_title].index[0]
        content_distances = similarity[content_movie_index]
        content_movies_list = sorted(list(enumerate(content_distances)), reverse=True, key=lambda x: x[1])[1:6]
        content_recommended_movies = [movies.iloc[i[0]].title for i in content_movies_list]

        # Collaborative filtering recommendations (Item-Item)
        collab_movie_id = movies[movies['title'] == movie_title]['id'].values[0]
        similarity_scores = item_similarities[collab_movie_id]
        similar_movie_indices = similarity_scores.argsort()[::-1][1:2]
        collab_recommended_movies = movies[movies['id'].isin(similar_movie_indices)]['title'].values



        # Combine and return recommendations
        hybrid_recommendations = []
        if content_recommended_movies:
            hybrid_recommendations.extend(content_recommended_movies)
        if collab_recommended_movies:
            hybrid_recommendations.extend(collab_recommended_movies)


        return hybrid_recommendations


    def fetch_poster(movie_id):
        api_key = '3cd07b4b483e1b50f7554420fda72adb'  # Replace with your TMDb API key
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
        data = response.json()
        if 'poster_path' in data:
            return "https://image.tmdb.org/t/p/original" + data['poster_path']
        else:
            return "https://example.com/default_poster.jpg"  # Replace with your default image URL or handle the error in your preferred way


    if st.button('Recommend'):
        recommendations = hybrid_recommender(selected_movie_name_hybrid)
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, name in enumerate(recommendations):
            with col1 if i == 0 else col2 if i == 1 else col3 if i == 2 else col4 if i == 3 else col5:
                st.text(name)
                movie_id = movies[movies['title'] == name]['id'].values[0]
                poster_url = fetch_poster(movie_id)
                st.image(poster_url)

