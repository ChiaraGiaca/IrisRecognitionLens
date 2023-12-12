import pickle
from glob import glob
from pandas.core.dtypes.cast import maybe_box_native
from torch.cuda.memory import max_memory_reserved
from tqdm import tqdm
import os
import random
import csv
import sys     
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
from scipy.spatial import distance
# appending the directory of utils.file_utils.py
# in the sys.path list
sys.path.append('/content/drive/MyDrive/Biometric Systems/iris-recognition/utils')
sys.path.append('/content/drive/MyDrive/Biometric Systems/iris-recognition')    
from file_utils import extract_user_sample_ids
from biometric_system import *
import numpy as np
from scipy.interpolate import make_interp_spline


REGISTERED_USERS = "../data_iiitd/system_database/registered_users"
UNKNOWN_USERS = "../data_iiitd/system_database/unknown_users"
MODEL_PATH = "iris_recognition_trained_model-2.pt"

#function to elaborate and plot the CMC curve at different ranks
def cmc_score(cmc_diz):
  max_rank = len(cmc_diz[0][1]) 
  y = []
  ranks = [0, 4, 9, 14, 19, 24, 29]
  for rank in range(max_rank):
    controller = 0
    for tupla in cmc_diz:
      if tupla[0] in tupla[1][:rank+1]:
        controller += 1
    
    y.append(controller/len(cmc_diz))

  X_Y_Spline = make_interp_spline(range(1,max_rank+1), y)
  X_ = np.linspace(1, max_rank, 500)
  Y_ = X_Y_Spline(X_)
  plt.plot(X_,Y_)
  plt.xlabel("Rank")
  plt.ylabel("Probability of identification")
  plt.show()

  for i in range(len(ranks)):
    print("\nTrue Positive Identification Rate at rank ", ranks[i] + 1, ": ", y[ranks[i]])

#Function to compute Multiclass 1vR ROC, AUC and EER, elaborating False Positive Rate and True Positive Rate, and plot the ROC CURVE
def roc_curve(roc_diz, max_t):
  max_threshold=max_t
  roc_auc={'fpr':[],'tpr':[],'auc':[]}

  labels=np.array(roc_diz[:,0])
  lab_outputs=roc_diz[0,1]
  outputs_l=roc_diz[:,2]
  outputs=[]
  for l in outputs_l:
    pro_l=[]
    for p in l:
      pro_l.append(p)
    outputs.append(pro_l)
  outputs=np.array(outputs)
  tholds=random.choice(outputs)
  
  for i in range(len(labels)):
    classe = labels[i]
    probabilities=outputs[:,lab_outputs.index(classe)]
    labB=[1 if x==classe else 0 for x in labels]
    nn_tpr, nn_fpr = get_all_roc_coordinates(labB, probabilities, classe,max_threshold,tholds)
    auc = metrics.roc_auc_score(labB, probabilities)

    roc_auc['fpr']+=[nn_fpr]
    roc_auc['tpr']+=[nn_tpr]
    roc_auc['auc'].append(auc)
 
  roc_fpr_mean=np.mean(np.array(roc_auc['fpr']),axis=0)
  roc_tpr_mean=np.mean(np.array(roc_auc['tpr']),axis=0)
  
  frr_mean=1 - roc_tpr_mean
  eer = roc_fpr_mean[np.nanargmin(np.absolute(frr_mean - roc_fpr_mean))]
  plot_roc_curve(roc_tpr_mean,roc_fpr_mean,frr_mean, tholds)
  thresholds=sorted(np.linspace(0,max_threshold,1000),reverse=True)
  maxv=10
  pos=0
  for i in range(len(thresholds)):
    p=(roc_fpr_mean[i],roc_tpr_mean[i])
    dist = distance.euclidean((0,1), p)
    if dist < maxv:
      maxv = dist
      pos=i
  print("Optimal threshold for verification: ",thresholds[pos])

 
  print("AUC: ",np.mean(roc_auc['auc']))
  print("EER: ",eer)

#subfunction that calculates FPRs and TPRs for the given class
def get_all_roc_coordinates(y_real, y_proba, class_id,max_threshold,tholds):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a threshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    
    tpr_list = [0]
    fpr_list = [0]

    
    #in order to have the tpr always increasing
    #thresholds=sorted(y_proba,reverse=True)
    thresholds=sorted(np.linspace(0,max_threshold,1000),reverse=True)
    for t in thresholds:
        threshold = t
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list

#subfunction that elaborates the specific TPRs and FPRs with binarized labels for the given class
def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations

    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes

    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''


    cm = confusion_matrix(y_real, y_pred)
    """disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[e for e in range(2)])
    disp.plot()
    plt.show()"""
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]


    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate

    return tpr, fpr

#function that plots the final averaged ROC curve
def plot_roc_curve(tpr, fpr, fnr, tholds, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).

    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''


    if ax == None:
        plt.figure(figsize = (5, 5))
        ax = plt.axes()

    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.title("ROC Curve")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("FAR - False Positive Rate")
    plt.ylabel("1-FRR - True Positive Rate")
    plt.show()

#function that plots the istograms with the accuracy of splitted "colored", "transparent" and "normal" probs 
def plot_correct_rates(rates):
  #I create a dictionary containing for every class a counter in which the first element
  #corresponds to the correct predictions of this type, and the second to the total
  
  df = pd.DataFrame.from_dict(rates, orient="index")
  display(df)
  fig, axes = plt.subplots(figsize=(20,10), dpi=100)
  plt.bar(df.index, height=df['correct_count']/df['total_count'])
  plt.title('Correct Prediction rates for each class')
  plt.show()
  print()

#function that takes a path of an image and, applying the model, tris to identify its class or verify the given one
def run_classification_(image_path: str, mode: str, user_id: str = None):
    image = Image(image_path=image_path)

    try:
        image.find_iris_and_pupil()
    except ImageProcessingException:
        return RunResults.PROCESSING_FAILURE, 0, 0, 0, 0
    
    iris = normalize_iris(image)
    iris.pupil = image.pupil
    iris.iris = image.iris

    # Save the normalized image to a temporary file for easier use with
    # the trained network
    create_empty_dir("tmp")
    iris_hash = hashlib.sha1(image_path.encode()).hexdigest()
    iris_path = f"tmp/{iris_hash}.jpg"
    iris.save(iris_path)

    # Get the classifier's prediction
    predicted_user, sorted_labels, sorted_probs, labels, probs = classifier.classify_single_image(iris_path)
    if mode == Mode.IDENTIFY:
        if predicted_user == User.UNKNOWN:
            run_result = RunResults.IDENTIFICATION_FAILURE
        else:
            run_result = RunResults.IDENTIFICATION_SUCCESS
    else: 
        if predicted_user == User.UNKNOWN:
            run_result = RunResults.VERIFICATION_FAILURE_USER_UNKNOWN
        else:
            if predicted_user == user_id:
                run_result = RunResults.VERIFICATION_SUCCESS
            else:
                run_result = RunResults.VERIFICATION_FAILURE_USER_MISMATCH
   
    # Remove temporary files
    remove(iris_path)

    return run_result, sorted_labels, sorted_probs, labels, probs


# User IDs of identified and verified users
stats = {
    "registered": {
        "identified": [],
        "not_identified": [],
        "verified": [],
        "not_verified": [],
        "accepted_probabilities": [],
        "rejected_probabilities": []
    },
}

classes_predictions_rate = {
      'Colored': {
        'correct_count': 0, 
        'total_count': 0
      },
      'Normal': {
        'correct_count': 0, 
        'total_count': 0
      },
      'Transparent': {
        'correct_count': 0, 
        'total_count': 0
      }, 
    }

cmc_diz = []
roc_diz=[]
colors = {}
with open('Description_IIITD-CLI.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  for row in reader:
    colors[row['ID']] = row['Ciba Vision Lens Color']

#here the choice of how many sample to use for the evaluation 

registered_user_paths = sorted(glob(REGISTERED_USERS + "/*"))
random.shuffle(registered_user_paths)
registered_user_paths = registered_user_paths[:1000]

# Load trained classifier
classifier = IrisClassifier(load_from_checkpoint=True,
                            checkpoint_file=MODEL_PATH)
max_t= 0
for user_path in tqdm(registered_user_paths, desc="Registered users"):
    user_id, type_img, _ = extract_user_sample_ids(user_path)
    #Run identification
    identification_result, sorted_labels, sorted_probs, labels, probs = run_classification_(user_path, Mode.IDENTIFY)
    
    indx_label
    if sorted_labels != 0:
      if user_id == sorted_labels[0]:
        classes_predictions_rate[type_img]['correct_count'] += 1
      classes_predictions_rate[type_img]['total_count'] += 1
      max_t=max(max_t,sorted_probs[0])
      cmc_diz.append([user_id, sorted_labels, sorted_probs])
      roc_diz.append([user_id, labels, probs])
    
    if identification_result == RunResults.IDENTIFICATION_SUCCESS:
        stats["registered"]["identified"].append(user_path)
    else:
        stats["registered"]["not_identified"].append(user_path)
    
    # Run verification
    verification_result, sorted_labels_ver, sorted_probs_ver, _, _ = run_classification_(user_path, Mode.VERIFY,
                                                    user_id)
    if verification_result == RunResults.VERIFICATION_SUCCESS:
        stats["registered"]["verified"].append(user_path)
    else:
        #print("\nUser id: ",user_id, "\nSorted labels: ", sorted_labels_ver,"\nSorted probabilities: ", sorted_probs_ver,"\nUser Path: ", user_path, "\n")
        stats["registered"]["not_verified"].append(user_path)


#calling functions for evaluation metrics
cmc_score(np.array(cmc_diz))
roc_curve(np.array(roc_diz), max_t)
plot_correct_rates(classes_predictions_rate)



def get_unique_ids(paths):
    return set(extract_user_sample_ids(path)[0] for path in paths)
    
#print accuracies and identifies classes for identify and verify mode
print("REGISTERED USERS:")
identified = stats["registered"]["identified"]
identified_unique = get_unique_ids(identified)
identified_perc = len(identified) / len(registered_user_paths)
print(f"Identified: {len(identified_unique)} ({identified_perc:.2%})")

verified = stats["registered"]["verified"]
verified_unique = get_unique_ids(verified)
not_verified = stats["registered"]["not_verified"]
not_verified_unique = get_unique_ids(not_verified)
print("Total number of unique users: ", len(verified_unique.union(not_verified_unique)))
verified_perc = len(verified) / len(registered_user_paths)
print(f"Verified: {len(verified_unique)}/{len(verified_unique.union(not_verified_unique))} ({verified_perc:.2%})")
