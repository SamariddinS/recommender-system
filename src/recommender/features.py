import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

from collections import Counter
from itertools import combinations
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Features:
	def __init__(self) -> None:
		pass

	@staticmethod
	def convert_categorical(df_X, _X):
		values = np.array(df_X[_X])
		# integer encode
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(values)
		# binary encode
		onehot_encoder = OneHotEncoder(sparse_output=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		df_X = df_X.drop(columns=_X)
		for j in range(integer_encoded.max() + 1):
			df_X.insert(loc=j + 1,
						column=str(_X) + str(j + 1),
						value=onehot_encoded[:, j])
		return df_X

	@staticmethod
	def extract(df, df_user, alpha_coefs=[0.045], alpha_param=1682):
		# alpha_coefs = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]

		for alpha_coef in alpha_coefs:
			pairs = []
			grouped = df.groupby(['MID', 'rate'])

			for key, group in grouped:
				pairs.extend(list(combinations(group['UID'], 2)))

			counter = Counter(pairs)
			alpha = alpha_coef * alpha_param  # 1m = 3883, param*i_no
			edge_list = map(
				list,
				Counter(el for el in counter.elements()
						if counter[el] >= alpha).keys())
			G = nx.Graph()

			for el in edge_list:
				G.add_edge(el[0], el[1], weight=1)
				G.add_edge(el[0], el[0], weight=1)
				G.add_edge(el[1], el[1], weight=1)

			pr = nx.pagerank(G.to_directed())
			df_user['PR'] = df_user['UID'].map(pr)
			df_user['PR'] /= float(df_user['PR'].max())
			dc = nx.degree_centrality(G)
			df_user['CD'] = df_user['UID'].map(dc)
			df_user['CD'] /= float(df_user['CD'].max())
			cc = nx.closeness_centrality(G)
			df_user['CC'] = df_user['UID'].map(cc)
			df_user['CC'] /= float(df_user['CC'].max())
			bc = nx.betweenness_centrality(G)
			df_user['CB'] = df_user['UID'].map(bc)
			df_user['CB'] /= float(df_user['CB'].max())
			lc = nx.load_centrality(G)
			df_user['LC'] = df_user['UID'].map(lc)
			df_user['LC'] /= float(df_user['LC'].max())
			nd = nx.average_neighbor_degree(G, weight='weight')
			df_user['AND'] = df_user['UID'].map(nd)
			df_user['AND'] /= float(df_user['AND'].max())
			X_train = df_user.loc[:, df_user.columns[1:]]
			X_train.fillna(0, inplace=True)

			X_train.to_pickle(parent_dir+"/data/extracted_features/x_train_alpha(" + str(alpha_coef) +").pkl")

			return X_train


	def extract_user_feature(self, data_path, ratings, df_user, rating_sep, users_sep):
		if data_path is not None:
			ratings = pd.read_csv(parent_dir+data_path + ratings,
								sep=rating_sep,
								engine='python',
								names=['UID', 'MID', 'rate', 'time'])
			df_user = pd.read_csv(parent_dir+data_path + df_user,
								sep=users_sep,
								engine='python',
								names=['UID', 'age', 'gender', 'job', 'zip'])

		# User Features
		df_user = self.convert_categorical(df_user, 'job')
		df_user = self.convert_categorical(df_user, 'gender')
		df_user['bin'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 100],
								labels=['1', '2', '3', '4', '5', '6'])
		df_user['age'] = df_user['bin']

		df_user = df_user.drop(columns='bin')
		df_user = self.convert_categorical(df_user, 'age')
		df_user = df_user.drop(columns='zip')
		X_train = self.extract(ratings, df_user)

		return X_train


	def load_data(self, data_path="/data/dataset/ml-100k/", ratings="u1.base", users="u.user", test_data="u1.test", rating_sep='\t', users_sep='\\|'):
		train = np.loadtxt(parent_dir+data_path + ratings, skiprows=0, delimiter=rating_sep).astype("int32")
		test = np.loadtxt(parent_dir+data_path + test_data, skiprows=0, delimiter=rating_sep).astype("int32")
		total = np.concatenate((train, test), axis=0)

		X_train = self.extract_user_feature(data_path=data_path, ratings=ratings, df_user=users, rating_sep=rating_sep, users_sep=users_sep)

		# Prepar train data
		n_u = np.unique(total[:, 0]).size  # num of users
		n_m = np.unique(total[:, 1]).size  # num of movies
		n_train = train.shape[0]  # num of training ratings
		n_test = test.shape[0]  # num of test ratings

		train_r = np.zeros((n_m, n_u), dtype="float32")
		test_r = np.zeros((n_m, n_u), dtype="float32")

		for i in range(n_train):
			train_r[train[i, 1] - 1, train[i, 0] - 1] = train[i, 2]

		for i in range(n_test):
			test_r[test[i, 1] - 1, test[i, 0] - 1] = test[i, 2]

		train_m = np.greater(train_r, 1e-12).astype("float32")  # masks indicating non-zero entries

		# Append the movies in X_train to the end of the existing movies in train_r
		train_r = np.concatenate((train_r,  X_train.T), axis=0).astype('float32')

		# save the ndarray object to a file using pickle
		with open(parent_dir+"/data/train_data/train_r_(" + str(n_u) +").pkl", "wb") as f:
			pickle.dump(train_r, f)

		return n_m, n_u, train_r, train_m, test_r


	def load_features(alpha_coef="0.045"):
		X_train = pd.read_pickle(parent_dir+f"/data/extracted_features/" + "features_alpha({alpha_coef}).pkl").values.astype(float)

		return X_train

	def load_train_data(n_u):
		train_r = pd.read_pickle(parent_dir+f"/data/train_data/" + "train_r_({n_u}).pkl").values.astype('float32')

		return n_u, train_r