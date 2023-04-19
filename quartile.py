# degree one
# degree_one_users [utente, n_interaz]
import numpy as np
import pandas as pd

df = pd.read_csv('dataset.tsv', sep='\t', header=None)


degree_one_users = df.iloc[:, 0].value_counts()
degree_one_users = degree_one_users.sort_index()
users = degree_one_users.index.values
values = degree_one_users.values

degree_one_users = np.column_stack((degree_one_users.index.values, degree_one_users.values))
# back = back.sort_values(by=[0])

# user_groups = np.zeros(np.max(degree_one_users[:, 0]))
user_groups = np.column_stack((users, np.zeros(len(degree_one_users)))).astype(int)
q1 = np.quantile(degree_one_users[1], 0.25)
q2 = np.quantile(degree_one_users[1], 0.50)
q3 = np.quantile(degree_one_users[1], 0.75)
# degree_one_users = degree_one_users.to_numpy()
# degree_one_users = degree_one_users.values[:, :1]

user_groups[degree_one_users[:,1] <= q1, 1] = 0
user_groups[(degree_one_users[:,1] > q1) & (degree_one_users[:, 1] <= q2), 1] = 1
user_groups[(degree_one_users[:,1] > q2) & (degree_one_users[:,1] <= q3), 1] = 2
user_groups[(degree_one_users[:,1] > q3), 1] = 3
# col1, col2 = [], []
# for idx, u in enumerate(user_groups.tolist()):
#     col1.append(private_users[idx])
#     col2.append(int(u))
#     df = pd.DataFrame([], columns=[1, 2])
#     df[1] = pd.Series(col1)
#     df[2] = pd.Series(col2)
# df.to_csv(f'./data/{data}/users_deg_1.tsv', sep='\t', header=None, index=None)

pd.DataFrame(user_groups).to_csv('user_deg.tsv', sep='\t', header=None, index=None)