from typing import List
import numpy as np

import metrics


class Recommender:
	def __init__(self, model, n_m, n_u, train_r, train_m, test_r):
		self.model = model
		self.n_m = n_m
		self.n_u = n_u
		self.train_r = train_r
		self.train_m = train_m
		self.test_r = test_r
		self.predict = None

	def train(self):
		if self.train_r is None:
			raise ValueError("Data is not loaded")

		# Instantiate and train the model
		metrics = self.model.fit(self.train_r)
		print(metrics)

	def make_predict(self):
		if self.n_u is None:
			raise ValueError("Data is not loaded")
		# Recommend for all users
		self.predict = self.model.predict(np.arange(self.n_u))

	def evaluate(self, k=50):
		if self.test_r is None or self.predict is None:
			raise ValueError("Data is not loaded")

		recommended = self.model
		ground_truth = np.argsort(-self.test_r, axis=0)[:k, :].T.tolist()
		recommended = np.argsort(-self.predict, axis=0)[:k, :].T.tolist()
		random = np.random.randint(0, self.n_m, (self.n_u, k)).T.tolist()

		print(
			"Baseline (random):\t",
			metrics.mapk(ground_truth, random, k=k),
			f"\n:{type(self.model).__name__}\t\t",
			metrics.mapk(ground_truth, recommended, k=k),
		)

	def top_n(self, user_id: int, n: int) -> List[int]:
		if self.predict is None:
			raise ValueError("Data is not loaded")

		predicted_ratings = self.predict.T[user_id - 1]  # user_id starts from 1
		rated_movies = np.where(self.train_m[:, user_id - 1] > 0)[0]  # movies already rated by user
		unrated_movies = np.setdiff1d(np.arange(self.n_m), rated_movies)  # movies not rated by user
		predicted_ratings[rated_movies] = -np.inf  # set rated movies' rating to -inf, so they won't be recommended

		# get top-n recommended movie IDs
		top_n = predicted_ratings.argsort()[::-1][:n]
		top_n_movie_ids = [movie_id + 1 for movie_id in top_n if movie_id in unrated_movies]

		return top_n_movie_ids

	def load(self):
		if self.train_r is None:
			raise ValueError("Data is not loaded")
		self.model.load(self.train_r)