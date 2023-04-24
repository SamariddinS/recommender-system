from recommender.recommender import Recommender
from recommender.glocalk import GlocalK
from recommender.features import Features

if __name__ == "__main__":

    # Extracting features
    features = Features()
    n_m, n_u, train_r, train_m, test_r = features.load_data()

    # Train Model
    model = GlocalK()
    recommender = Recommender(model, n_m, n_u, train_r, train_m, test_r)
    # recommender.train()
    recommender.load()
    recommender.make_predict()
    recommender.evaluate()

    # Get top 5 recommendations for user user_id
    user_id = 10
    top_n_recommendations = recommender.top_n(user_id, n=5)

    print(f"Top 5 recommendations for user {user_id}: {top_n_recommendations}")
