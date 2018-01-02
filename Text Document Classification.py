import math
import numpy as np
import operator as op 
# test = "rt-test.txt"
# train = "rt-train.txt"
test = "fisher_test_2topic.txt"
train = "fisher_train_2topic.txt"
def getFile(file):
	with open(file) as f:

		text = list(f.readlines())
		
	text = [i.split(" ") for i in text]
	return text

# this will create dictionary with respect to a class
def makeDict(List, Class):
	Dict = {}
	for i in range(len(List)):
		new = [m.split(':') for  m in List[i]]
		if(int(new[0][0]) == Class ):
			for j in range(1, len(new)):
				if new[j][0] in Dict:
					Dict[new[j][0]] += int(new[j][1])
				else:
					Dict[new[j][0]] = int(new[j][1])
	if(Class == 1):
		print("Pos Vocab")
	else:
		print("Neg Vocab")
	print(len(Dict))
	return Dict

#this is making a dictionary of all words used regardless of class
def unique_vocab(List):
	Dict = {}
	for i in range(len(List)):
		new = [m.split(':') for  m in List[i]]
		for j in range(1, len(new)):
			if new[j][0] in Dict:
				Dict[new[j][0]] += int(new[j][1])
			else:
				Dict[new[j][0]] = int(new[j][1])

	print("Unique Vocab")
	print(len(Dict))
	return Dict

def Pclass(posrev,negrev,Class):
	if(Class > 0):
		Ppos = abs(math.log10(posrev/(posrev + negrev +.0)))
		return Ppos
	else:
		Pneg = abs(math.log10(negrev/(posrev + negrev + .0)))
		return Pneg

def multinomial_NB(Dict, key, vocab, Words_in_class):
	# Words_in_class will give the total words in positive case
	# key is the word for the probability
	# vocab will give unique words in a certain case
	if key in Dict:
		Prob = abs(math.log10((int(Dict[key]) + 1 + .0)/( Words_in_class + vocab )))
	else:
		Prob = abs(math.log10((1 + .0)/(Words_in_class + vocab)))
	return Prob
def multi_NB_no_word(vocab, Words_in_class):
	Prob = abs(math.log10((1 +.0)/(Words_in_class + vocab)))
	return Prob
def getClassProb(Dict, vocab, Words_in_class):
	# this will create the probability dictionary
	ProbDict = {}
	for words in Dict:
		ProbDict[words] = multinomial_NB(Dict,words,vocab,Words_in_class)
	return ProbDict
# vi = p(class) * pi( p(w1 } class)... p(wn|class))
# p(word|class) = (wordfrequency + 1) / (( total # word in class) + unique_vocabulary)
# p(word|class) = f + 1 \ nk + vocab

def TestData(ProbPos, ProbNeg, file, posClass,negClass, vocab, WICPos, WICNeg):
	# this function will create a list of the prediction of the sentences in test
	# ClassProb = probability of a given class ( + or -)
	# file = the testing data set
	# ProbPos = a dictionary filled where the value is the probability for the key for a certain class
	# ProbNeg = ^^
	# def multinomial_NB(Dict, key, vocab, Words_in_class):
	# def multi_NB_no_word(vocab, Words_in_class):
	TestResultPos = []
	TestResultNeg = []
	pos_Prob = posClass
	neg_Prob = negClass
	Actual_Result = []
	for line in range(len(file)):
		new = [k.split(':') for  k in file[line]]
		Actual_Result.append(int(new[0][0]))
		pos_Prob = posClass
		neg_Prob = negClass
		for word in range(1,len(new)):
			if new[word][0] in ProbPos:
				pos_Prob = pos_Prob * ProbPos[new[word][0]]
			else:
				pos_Prob = pos_Prob * multi_NB_no_word(vocab,WICPos)
			if new[word][0] in ProbNeg:
				neg_Prob = neg_Prob * ProbNeg[new[word][0]]
			else:
				neg_Prob = neg_Prob * multi_NB_no_word(vocab,WICNeg)
		TestResultPos.append(pos_Prob)
		TestResultNeg.append(neg_Prob)
	TestResult = []
	TestResult.append(TestResultPos)
	TestResult.append(TestResultNeg)
	TestResult.append(Actual_Result)
	return TestResult

def determineClassification(pos, neg):
	# this will tell if the review is overall predicted as positive or negative
	Result = []
	for i in range(len(pos)):
		if(pos[i] > neg[i]):
			Result.append(-1)
		else:
			Result.append(1)
	print("Result", len(Result))
	return Result

def CheckAccuracy(predict,Actual):
	correct = 0.00
	for i in range(len(Actual)):
		if(predict[i] == Actual[i]):
			correct += 1
	Accuracy = (correct + .0) / len(Actual)
	return Accuracy
def CMatrix( predict, Actual):
	True_pos = 0
	True_neg = 0
	False_pos = 0
	False_neg = 0
	for i in range(len(Actual)):
		if( predict[i] > 0 and Actual[i] < 0):
			False_pos += 1
		elif( predict[i] > 0 and Actual[i] > 0):
			True_pos += 1
		elif( predict[i] < 0 and Actual[i] > 0):
			False_neg += 1
		else:
			True_neg += 1

	matrix = np.zeros((2,2))
	print("True Negative, False Positive")
	matrix[0][0] = True_neg
	matrix[0][1] = False_pos
	matrix[1][0] = False_neg
	matrix[1][1] = True_pos
	print(matrix)
	print("False negative, True Positive")

# def WordLikelihod(Dict):
# 	i = 0
# 	words = []
# 	for keys in Dict:
# 		if( i < 9):
# 			words.append(keys)
# 		else:
# 			frequency = Dict[i]



print("Sentimental analysis of movie reviews")
test = getFile(test)
train = getFile(train)

# successfully created list to orderly store these words
# now create dictionary
pos = 1
neg = -1
# words in positive class and negative class
posWord = 0
negWord = 0
# for calculating the # of reviews for each respective
posrev = 0
negrev = 0


# Dict = {}
# print(train[0])

# new = [m.split(':') for  m in train[0]]
# print(new[0][0])
# var = int(new[0][0]) + 1
# print(len(new))
# print("Testing...")
# for i in range(1,len(new)):
# 	print(new[i][0])
# 	Dict[new[i][0]] = new[i][1]
def BernoulliBuildDict(List, Class):
	Dict = {}
	for i in range(len(List)):
		new = [m.split(':') for  m in List[i]]
		if(int(new[0][0]) == Class ):
			for j in range(1, len(new)):
				if new[j][0] in Dict:
					Dict[new[j][0]] += 1
				else:
					Dict[new[j][0]] = 1
	if(Class == 1):
		print("Pos Vocab")
	else:
		print("Neg Vocab")
	print(len(Dict))
	return Dict
def Bernoulli(Dict, Classdict,review ):
	# this dictionary will take a key and return a value in probability 
	BerDict = {}
	for key in Dict:
		if key in Classdict:
			BerDict[key] = (Classdict[key] + .0) / review
		else:
			BerDict[key] = 0
	return BerDict


def BernoulliDataTest( pos, neg, posprior,negprior, file):
	# if word is in review, multiply probability, if not multiply by not probability
	positive = []
	negative = []
	Actual_reveiew = []
	Result = []
	for line in range(len(file)):
		new = [k.split(':') for  k in file[line]]
		positive_probability = posprior
		negative_probabilty = negprior
		Actual_reveiew.append(int(new[0][0]))
		for key in pos:
			flag = 0
			for word in range(1,len(new)):
				if(new[word][0] == key):
					flag = 1
			if(flag == 1):
				# print(key)
				if(pos[key] != 0):
					positive_probability = positive_probability + abs(math.log10(pos[key]))
				if(neg[key] != 0):
					negative_probabilty = negative_probabilty + abs(math.log10(neg[key]))
			else:	
				if(pos[key] != 1):
					positive_probability = positive_probability + abs(math.log10(1.0 - pos[key]))
				if(neg[key] != 1):
					negative_probabilty = negative_probabilty + abs(math.log10(1.0 - neg[key]))
		positive.append(positive_probability)
		negative.append(negative_probabilty)
	Result.append(positive)
	Result.append(negative)
	Result.append(Actual_reveiew)
	return Result

def getBernoulliResult(pos, neg):
	result = []
	for i in range(len(pos)):
		result.append(pos[i]/(pos[i] + neg[i]))
	return result

def BernoulliAccuracy(predicted, Actual):
	correct = 0
	for i in range(len(predicted)):
		if(predicted[i] <= .5):
			if(int(Actual[i]) == 1):
				correct += 1
	Accuracy = (correct + .0)/ len(Actual)
	return Accuracy

# def ZiweiBernoulliCalc(pos, neg):



# Calculating positive articles and # of total words
for Class in range(len(train)):
	if(train[Class][0] == '1'):
		posrev += 1
		new = [m.split(':') for  m in train[Class]]
		for i in range(1, len(new)):
			posWord += int(new[i][1])
	elif(train[Class][0] == '-1'):
		negrev += 1
		new = [m.split(':') for  m in train[Class]]
		for i in range(1, len(new)):
			negWord += int(new[i][1])
print("")
# a few print statements for key information
print("positive word","   negative word")
print(posWord,"          ", negWord)
print(" ")
print("reviews")
print(posrev,negrev)
print(" ")
print("Sum of All words in reviews")
print(posWord + negWord)
print(" ")
PosClass = Pclass(posrev,negrev,1)
print("Log Probability of positive review")
print(PosClass)
print(" ")
NegClass = Pclass(posrev,negrev,-1)
print("Log Probability of negative review")
print(NegClass)

# positive vocab dictionary and negative vocab dictionary
TrainPos = makeDict(train, 1)
TrainNeg = makeDict(train, -1)
# implement MULTINOMIAL naive bayes now

#vocab is the number of unique words in the training data
vocab_dict = unique_vocab(train)
vocab = len(vocab_dict)

Probpos = getClassProb(TrainPos, vocab,posWord )
Probneg = getClassProb(TrainNeg, vocab, negWord)

# ok finished building probability of world dictionary
TestResult = TestData(Probpos, Probneg, test,PosClass,NegClass, vocab, posWord, negWord )
positive_result = TestResult[0]
negative_result = TestResult[1]
Actual_Result = TestResult[2]
Classification = determineClassification(positive_result,negative_result)
Accuracy = CheckAccuracy(Classification, TestResult[2])
# print accuracy
print("Accuracy : ",Accuracy * 100,"%")
CMatrix(Classification, TestResult[2])
print("Highest Likelihood for Positive:")
sorted_pos = sorted(Probpos.items(), key = op.itemgetter(1))
print(sorted_pos[-10:])
print("Highest Likelihood for Negative:")
sorted_neg = sorted(Probneg.items(), key = op.itemgetter(1))
print(sorted_neg[-10:])

odds_ratio = {}
for key in Probpos:
	if key in Probneg:
		odds_ratio[key] = np.log(Probpos[key]) - np.log(Probneg[key])
sorted_odds = sorted(odds_ratio.items(), key= op.itemgetter(1))
print("10 highest odds ratio words")
print(sorted_odds[-10:])

#Bernoulli's
# so basically see how many times a word appears in a classification and then divide the the total number of times it occured
# these are class priors


# priorpos = PosClass
# priorneg = NegClass
# retrieve bernoulli probability dictionaries
# send in the unique vocab words, training
# BTrainPos = BernoulliBuildDict(train,1)
# BTrainNeg = BernoulliBuildDict(train, -1)
# Bernoulli_Dict_pos = Bernoulli(vocab_dict, BTrainPos, posrev)
# Bernoulli_Dict_neg = Bernoulli(vocab_dict, BTrainNeg, negrev)
# Result = BernoulliDataTest(Bernoulli_Dict_pos,Bernoulli_Dict_neg, priorpos,priorneg, test)
# PosResult = Result[0]
# NegResult = Result[1]
# ActResult = Result[2]

# result = getBernoulliResult(PosResult,NegResult)
# Accuracy = BernoulliAccuracy(result,ActResult)
# print(Accuracy)





