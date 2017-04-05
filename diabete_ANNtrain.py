#train an ANN clssifier on disbete dataset
#considering categorical and continous features


import pandas as pd
import tensorflow as tf


TRAINING="diabetic_data_clean_train.csv"
TEST="diabetic_data_clean_test.csv"

#read data
COLUMNS=[ "race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id","time_in_hospital",  "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses", "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose","miglitol", "troglitazone", "tolazamide",  "insulin", "glyburide-metformin", "glipizide-metformin",  "metformin-pioglitazone", "change", "diabetesMed", "readmitted"]
CATEGORICAL_COLUMNS=["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "admission_source_id", "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",  "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose","miglitol", "troglitazone", "tolazamide",  "insulin", "glyburide-metformin", "glipizide-metformin",  "metformin-pioglitazone", "change", "diabetesMed"]
CONTINUOUS_COLUMNS=[ "time_in_hospital",  "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]

train=pd.read_csv(TRAINING, names=COLUMNS)
test=pd.read_csv(TEST, names=COLUMNS)
train = train.dropna(how='any', axis=0)
test = test.dropna(how='any', axis=0)
LABEL_COLUMN="label"
train[LABEL_COLUMN]=(train["readmitted"].apply(lambda x: "<30" in x)).astype(int)
test[LABEL_COLUMN]=(test["readmitted"].apply(lambda x: "<30" in x)).astype(int)


#continous features
time_in_hospital = tf.contrib.layers.real_valued_column("time_in_hospital")
num_lab_procedures = tf.contrib.layers.real_valued_column("num_lab_procedures")
num_procedures = tf.contrib.layers.real_valued_column("num_procedures")
num_medications= tf.contrib.layers.real_valued_column("num_medications")
number_outpatient = tf.contrib.layers.real_valued_column("number_outpatient")
number_emergency = tf.contrib.layers.real_valued_column("number_emergency")
number_inpatient = tf.contrib.layers.real_valued_column("number_inpatient")
number_diagnoses = tf.contrib.layers.real_valued_column("number_diagnoses")

#categorical features
race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",keys=["Female", "Male"])
age = tf.contrib.layers.sparse_column_with_hash_bucket("age", hash_bucket_size=100)
admission_type_id =tf.contrib.layers.sparse_column_with_hash_bucket("admission_type_id", hash_bucket_size=100)
discharge_disposition_id = tf.contrib.layers.sparse_column_with_hash_bucket("discharge_disposition_id", hash_bucket_size=100)
admission_source_id = tf.contrib.layers.sparse_column_with_hash_bucket("admission_source_id", hash_bucket_size=100)
max_glu_serum = tf.contrib.layers.sparse_column_with_hash_bucket("max_glu_serum", hash_bucket_size=100)
A1Cresult = tf.contrib.layers.sparse_column_with_hash_bucket("A1Cresult", hash_bucket_size=100)
metformin= tf.contrib.layers.sparse_column_with_hash_bucket("metformin", hash_bucket_size=100)
repaglinide = tf.contrib.layers.sparse_column_with_hash_bucket("repaglinide", hash_bucket_size=100)
nateglinide = tf.contrib.layers.sparse_column_with_hash_bucket("nateglinide", hash_bucket_size=100)
chlorpropamide = tf.contrib.layers.sparse_column_with_hash_bucket("chlorpropamide", hash_bucket_size=100)
glimepiride = tf.contrib.layers.sparse_column_with_hash_bucket("glimepiride", hash_bucket_size=100)
#acetohexamide = tf.contrib.layers.sparse_column_with_hash_bucket("acetohexamide", hash_bucket_size=100)
glipizide = tf.contrib.layers.sparse_column_with_hash_bucket("glipizide", hash_bucket_size=100)
glyburide= tf.contrib.layers.sparse_column_with_hash_bucket("glyburide", hash_bucket_size=100)
tolbutamide = tf.contrib.layers.sparse_column_with_hash_bucket("tolbutamide", hash_bucket_size=100)
pioglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("pioglitazone", hash_bucket_size=100)
rosiglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("rosiglitazone", hash_bucket_size=100)
acarbose = tf.contrib.layers.sparse_column_with_hash_bucket("acarbose", hash_bucket_size=100)
miglitol = tf.contrib.layers.sparse_column_with_hash_bucket("miglitol", hash_bucket_size=100)
troglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("troglitazone", hash_bucket_size=100)
tolazamide = tf.contrib.layers.sparse_column_with_hash_bucket("tolazamide", hash_bucket_size=100)
#examide= tf.contrib.layers.sparse_column_with_hash_bucket("examide", hash_bucket_size=100)
#citoglipton= tf.contrib.layers.sparse_column_with_hash_bucket("citoglipton", hash_bucket_size=100)
insulin = tf.contrib.layers.sparse_column_with_hash_bucket("insulin", hash_bucket_size=100)
glyburide_metformin = tf.contrib.layers.sparse_column_with_hash_bucket("glyburide-metformin", hash_bucket_size=100)
glipizide_metformin = tf.contrib.layers.sparse_column_with_hash_bucket("glipizide-metformin", hash_bucket_size=100)
#glimepiride_pioglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("glimepiride-pioglitazone", hash_bucket_size=100)
#metformin_rosiglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("metformin-rosiglitazone", hash_bucket_size=100)
metformin_pioglitazone = tf.contrib.layers.sparse_column_with_hash_bucket("metformin-pioglitazone", hash_bucket_size=100)
change = tf.contrib.layers.sparse_column_with_hash_bucket("change", hash_bucket_size=100)
diabetesMed = tf.contrib.layers.sparse_column_with_hash_bucket("diabetesMed", hash_bucket_size=100)
diag_1 = tf.contrib.layers.sparse_column_with_hash_bucket("diag_1", hash_bucket_size=100)
diag_2 = tf.contrib.layers.sparse_column_with_hash_bucket("diag_2", hash_bucket_size=100)
diag_3 = tf.contrib.layers.sparse_column_with_hash_bucket("diag_3", hash_bucket_size=100)


#build deep columns for continous and categorical features
deep_columns = [
    tf.contrib.layers.embedding_column(race, dimension=8),
    tf.contrib.layers.embedding_column(gender, dimension=8),
    tf.contrib.layers.embedding_column(age, dimension=8),
    tf.contrib.layers.embedding_column(admission_type_id, dimension=20),
    tf.contrib.layers.embedding_column(discharge_disposition_id,dimension=20),
    tf.contrib.layers.embedding_column(admission_source_id , dimension=20),
    #tf.contrib.layers.embedding_column(payer_code, dimension=8),
    #tf.contrib.layers.embedding_column(medical_specialty , dimension=8),
    tf.contrib.layers.embedding_column(diag_1, dimension=8),
	tf.contrib.layers.embedding_column(diag_2, dimension=8),
	tf.contrib.layers.embedding_column(diag_3, dimension=8),
    tf.contrib.layers.embedding_column(max_glu_serum, dimension=8),
    tf.contrib.layers.embedding_column(A1Cresult, dimension=8),
    tf.contrib.layers.embedding_column(metformin, dimension=8),
    tf.contrib.layers.embedding_column(repaglinide, dimension=8),
    tf.contrib.layers.embedding_column(nateglinide, dimension=8),
    tf.contrib.layers.embedding_column(chlorpropamide , dimension=8),
    tf.contrib.layers.embedding_column(glimepiride  , dimension=8),
    #tf.contrib.layers.embedding_column(acetohexamide , dimension=8),
    tf.contrib.layers.embedding_column(glipizide , dimension=8),
    tf.contrib.layers.embedding_column(glyburide , dimension=8),
    tf.contrib.layers.embedding_column(tolbutamide , dimension=8),
    tf.contrib.layers.embedding_column(pioglitazone , dimension=8),
    tf.contrib.layers.embedding_column(rosiglitazone, dimension=8),
    tf.contrib.layers.embedding_column(acarbose  , dimension=8),
    tf.contrib.layers.embedding_column(miglitol , dimension=8),
    tf.contrib.layers.embedding_column(troglitazone , dimension=8),
    tf.contrib.layers.embedding_column(tolazamide, dimension=8),
    #tf.contrib.layers.embedding_column(examide, dimension=8),
    #tf.contrib.layers.embedding_column(citoglipton, dimension=8),
    tf.contrib.layers.embedding_column(insulin , dimension=8),
    tf.contrib.layers.embedding_column(glyburide_metformin, dimension=8),
    tf.contrib.layers.embedding_column(glipizide_metformin, dimension=8),
    #tf.contrib.layers.embedding_column(glimepiride_pioglitazone, dimension=8),
	#tf.contrib.layers.embedding_column(metformin_rosiglitazone, dimension=8),
	tf.contrib.layers.embedding_column(metformin_pioglitazone, dimension=8),
	tf.contrib.layers.embedding_column(change, dimension=8),
	tf.contrib.layers.embedding_column(diabetesMed , dimension=8),


	time_in_hospital,
	num_lab_procedures,
	num_procedures,
	num_medications,
	number_outpatient,
	number_emergency,
	number_inpatient,
	number_diagnoses,

]

def input_fn(df):
	"""Input builder function."""
	# Creates a dictionary mapping from each continuous feature column name (k) to
	# the values of that column stored in a constant Tensor.
	continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
	# Creates a dictionary mapping from each categorical feature column name (k)
	# to the values of that column stored in a tf.SparseTensor.
	categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],values=df[k].values,shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
	# Merges the two dictionaries into one.
	feature_cols = dict(continuous_cols)
	feature_cols.update(categorical_cols)
	# Converts the label column into a constant Tensor.
	label = tf.constant(df[LABEL_COLUMN].values)
	# Returns the feature columns and the label.
	return feature_cols, label


#build up ANN classifier

classifier = tf.contrib.learn.DNNClassifier(hidden_units=[100, 50],feature_columns=deep_columns, n_classes=3)
classifier.fit(input_fn=lambda: input_fn(train),  steps=500)

#write to file
f = open('output.txt', 'w')
for i in range(0, len(test)-1):
	results = classifier.evaluate(input_fn=lambda: input_fn(test.iloc[[i]]), steps=1)
	for key in sorted(results):
		#print("%s: %s" % (key, results[key]))
		if key =='accuracy':
			f.write(str(results[key]))
			f.write("\n")









