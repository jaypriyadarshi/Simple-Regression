import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import itertools

def prep_data():
	#load data
	data = load_boston()

	#generate trian and test data
	test_mask = [i for i in range(0, 506, 7)]
	train_mask = []
	for i in range(data.data.shape[0]):
		if i not in test_mask:
			train_mask.append(i)

	train_feats = data.data[train_mask]
	train_targets = data.target[train_mask]
	test_feats = data.data[test_mask]
	test_targets = data.target[test_mask]

	return train_feats, train_targets, test_feats, test_targets

def show_hist(train_feats):
	num_feats = train_feats.shape[1]
	for i in range(num_feats):
		plt.hist(train_feats[:,i], bins=10)
		plt.show()

def pearson_corr(train_feats, train_targets):
	num_feats = train_feats.shape[1]
	p_corr = []
	for i in range(num_feats):
		p_corr.append(np.cov(train_feats[:,i], train_targets)[0,1] / (np.std(train_feats[:,i], axis=0) * np.std(train_targets, axis=0)))
	return np.array(p_corr, dtype=np.float64)

def normalize_train(feats):
	mean = np.mean(feats, axis=0)
	std = np.std(feats, axis=0)
	return (feats - mean) / std, mean, std

def normalize_test(feats, mean, std):
	return (feats - mean) / std

def bias_trick(feats):
	#append 1 to each feature for bias term, so we need to perform only one matrix multiplication to get tagert
	return np.hstack((np.ones((feats.shape[0], 1), dtype=feats.dtype), feats))

def train(train_feats, train_targets):
	#use normal equation: weights = inv(X.T . X) . X.T . y
	return np.dot(np.dot(np.linalg.pinv(np.dot(train_feats.T, train_feats)), train_feats.T), train_targets)

def train_w_reg(train_feats, train_targets, reg):
	#use normal equation: weights = inv(X.T . X + reg_strength * Identity_mat) . X.T . y
	return np.dot(np.dot(np.linalg.pinv(np.dot(train_feats.T, train_feats) + (reg * np.identity(train_feats.shape[1], dtype=train_feats.dtype))), train_feats.T), train_targets)

def predict(feats, weights):
	return np.dot(feats, weights)

#compute mean squared error
def compute_mse(target, pred):
	num_examples = target.shape[0]
	sq_error = np.sum(np.square(target - pred), axis=0)
	return sq_error / num_examples

def linear_reg(train_feats, train_targets, test_feats, test_targets, print_results=True):
	#train the model
	weights = train(train_feats, train_targets)
	#predict train and test targets
	pred_train = predict(train_feats, weights)
	pred_test = predict(test_feats, weights)
	if print_results:
		print '-' * 20
		print "Linear Regression"
		print '-' * 20
		print "Training Data MSE: ", compute_mse(train_targets, pred_train)
		print "Test Data MSE: ", compute_mse(test_targets, pred_test)
		print
	else:
		return pred_train, pred_test

def ridge_reg(train_feats, train_targets, test_feats, test_targets, reg, cv=False):
	weights = train_w_reg(train_feats, train_targets, reg)
	#predict train and test targets
	pred_train = predict(train_feats, weights)
	pred_test = predict(test_feats, weights)
	if cv:
		mse = compute_mse(test_targets, pred_test)
		return mse
	else:
		print '-' * 20
		print "Ridge Regression with lambda = ", reg
		print '-' * 20
		print "Training Data MSE: ", compute_mse(train_targets, pred_train)
		print "Test Data MSE: ", compute_mse(test_targets, pred_test)

def cross_validate(train_feats_folds, train_target_folds ,num_folds, reg):
	#stores result for each fold
	accuracies = []
	for i in range(num_folds):
		cv_train_feats = np.vstack(train_feats_folds[0:i] + train_feats_folds[i+1:])
		cv_test_feats = train_feats_fold[i]
		cv_train_targets = np.hstack(train_target_folds[0:i] + train_target_folds[i+1:])
		cv_test_targets = train_target_folds[i]
		accuracies.append(ridge_reg(cv_train_feats, cv_train_targets, cv_test_feats, cv_test_targets, reg, cv=True))
	#compute average mean squared error	
	avg_mse = np.mean(accuracies)
	print "Lambda: %f Average Test Fold MSE: %f" % (reg, avg_mse)
	return avg_mse


#load data
train_feats, train_targets, test_feats, test_targets = prep_data()
show_hist(train_feats)
#normalize train and test feats
pearson_correlation = pearson_corr(train_feats, train_targets)
print "Pearson Correlation: ", pearson_correlation
print
train_feats, mean, std = normalize_train(train_feats)
test_feats = normalize_test(test_feats, mean, std)
#add bias term to features
train_feats = bias_trick(train_feats)
test_feats = bias_trick(test_feats)

#Linear Regression
linear_reg(train_feats, train_targets, test_feats, test_targets)

#Ridge Regression
reg_choices = [0.01, 0.1, 1.0]
#train the model
for reg in reg_choices:
	ridge_reg(train_feats, train_targets, test_feats, test_targets, reg)
print

#perform cross validation to find the best regularization strength
num_folds = 10
print '#' * 30
print 'Cross Validation with %d folds' % num_folds
print '#' * 30

train_feats_fold = []
train_target_folds = []
#split arrray into folds
train_feats_fold = np.array_split(train_feats, num_folds)
train_target_folds = np.array_split(train_targets, num_folds) 

reg_choices = np.linspace(0.0001, 10, num=1000)

#dictionary storing average mean squared error(mse) for each value of reg, (not doing this right now, it will store a list of len = num_folds, for each test_cv)
reg_mse = {}
#initialize best regularization strength and min avg mean squared error over all folds 
best_reg = reg_choices[0]
min_mse = float("inf")

for reg in reg_choices:
	reg_mse[reg] = cross_validate(train_feats_fold, train_target_folds, num_folds, reg)
	if reg_mse[reg] < min_mse:
		best_reg = reg
		min_mse = reg_mse[reg]

print 
print '#' * 30
print 'Best CV lambda: %f with average cross validation mse: %f' % (best_reg, reg_mse[best_reg])
ridge_reg(train_feats, train_targets, test_feats, test_targets, best_reg)
print '#' * 30
print

#second part of the assignment

#part a
print '#' * 20
print '3.3(a)'
print '#' * 20
#select 4 features with highest correlationwith the target and add 1 to the index as a new bias column was added to the orginial train_feats
top_four = (np.argsort(np.absolute(pearson_correlation))[-4:]) + 1
print "Feature indices of the 4 selected features (1st coulmn is bias, so all indices are shifted by 1)", top_four
new_train_feats = train_feats[:, top_four]
new_test_feats = test_feats[:, top_four]
new_train_feats = bias_trick(new_train_feats)
new_test_feats = bias_trick(new_test_feats)
#train the model
linear_reg(new_train_feats, train_targets, new_test_feats, test_targets)

#part b
print '#' * 20
print '3.3(b)'
print '#' * 20
feats_idx = []
residue = np.copy(train_targets)
while len(feats_idx) != 4:
	correlation = pearson_corr(train_feats[:,1:], residue)
	best_corr = (np.argsort(np.absolute(pearson_correlation))) + 1
	for i in range(best_corr.shape[0]):
		if best_corr[-i] not in feats_idx:
			feats_idx.append(best_corr[-i])
			break
	new_train_feats = bias_trick(train_feats[:,np.array(feats_idx)])
	new_test_feats = bias_trick(test_feats[:,np.array(feats_idx)])
	curr_train_pred, curr_test_pred = linear_reg(new_train_feats, train_targets, new_test_feats, test_targets, print_results=False)
	residue = np.absolute(train_targets - curr_train_pred)
print "Feature indices of the 4 selected features (1st coulmn is bias, so all indices are shifted by 1)", feats_idx
new_train_feats = bias_trick(train_feats[:,np.array(feats_idx)])
new_test_feats = bias_trick(test_feats[:,np.array(feats_idx)])
linear_reg(new_train_feats, train_targets, new_test_feats, test_targets)

#Brute search for 4 features
print '#' * 20
print 'Brute Force Search'
print '#' * 20
all_feat_combinations = []
#as the first column is bias
idx = [i+1 for i in range(13)]
for combination in itertools.combinations(idx, 4):
	all_feat_combinations.append(combination)

best_score = float("inf")

for combination in all_feat_combinations:
	new_feats_idx = np.array(list(combination))
	new_train_feats = bias_trick(train_feats[:,new_feats_idx])
	new_test_feats = bias_trick(test_feats[:,new_feats_idx])
	curr_train_pred, curr_test_pred = linear_reg(new_train_feats, train_targets, new_test_feats, test_targets, print_results=False)
	score = compute_mse(train_targets, curr_train_pred)
	if score < best_score:
		best_score = score
		best_feats = np.copy(new_feats_idx)

print "Feature indices of the best 4 features (1st column is bias, so all indices are shifted by 1)", best_feats
new_train_feats = bias_trick(train_feats[:,best_feats])
new_test_feats = bias_trick(test_feats[:,best_feats])
linear_reg(new_train_feats, train_targets, new_test_feats, test_targets)

#last part  - polynomial feature expansion
print '#' * 30
print 'Polynomial Feature Expansion'
print '#' * 30
poly_train_feats = np.ones((train_feats.shape[0], 1), dtype=train_feats.dtype)
poly_test_feats = np.ones((test_feats.shape[0], 1), dtype=test_feats.dtype)
#bias is still in the first column, skipping that column
for i in range(train_feats.shape[1]-1):
	feat1 = train_feats[:,i+1]
	new_feat = np.reshape(feat1, (feat1.shape[0], 1))
	poly_train_feats = np.hstack((poly_train_feats, new_feat))
	for j in range(i+1,train_feats.shape[1]):
		feat2 = train_feats[:,j]
		new_feat = np.reshape(feat1 * feat2, (feat1.shape[0], 1))
		poly_train_feats = np.hstack((poly_train_feats, new_feat))

for i in range(test_feats.shape[1]-1):
	feat1 = test_feats[:,i+1]
	new_feat = np.reshape(feat1, (feat1.shape[0], 1))
	poly_test_feats = np.hstack((poly_test_feats, new_feat))
	for j in range(i+1,test_feats.shape[1]):
		feat2 = test_feats[:,j]
		new_feat = np.reshape(feat1 * feat2, (feat1.shape[0], 1))
		poly_test_feats = np.hstack((poly_test_feats, new_feat))

#remove the first column of 1s, inserted initially just to make life easier while performing hstack
poly_train_feats = poly_train_feats[:,1:]
poly_test_feats = poly_test_feats[:,1:]
#normalize features
poly_train_feats, mean, std = normalize_train(poly_train_feats)
poly_test_feats = normalize_test(poly_test_feats, mean, std)
#add bias column
poly_train_feats = bias_trick(poly_train_feats)
poly_test_feats = bias_trick(poly_test_feats)
#perform linear regression on new higher order features
linear_reg(poly_train_feats, train_targets, poly_test_feats, test_targets)









