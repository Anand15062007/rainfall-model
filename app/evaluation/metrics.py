"""Classification, event and regression scores with safe undefined-metric handling."""
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score

def classification_metrics(y_true,y_pred,y_prob=None):
    y=np.asarray(y_true); p=np.asarray(y_pred); tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel(); h=tp+fn; total=tp+fp+fn+tn
    out={"accuracy":accuracy_score(y,p),"precision":precision_score(y,p,zero_division=0),"recall":recall_score(y,p,zero_division=0),"f1":f1_score(y,p,zero_division=0),
      "roc_auc":roc_auc_score(y,y_prob) if y_prob is not None and len(np.unique(y))>1 else float("nan"),"pr_auc":average_precision_score(y,y_prob) if y_prob is not None and len(np.unique(y))>1 else float("nan"),
      "pod":tp/h if h else 0.,"far":fp/(tp+fp) if tp+fp else 0.,"csi":tp/(tp+fp+fn) if tp+fp+fn else 0.,"ets":_ets(tp,fp,fn,tn),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}
    return out

def _ets(tp,fp,fn,tn):
    n=tp+fp+fn+tn; random_hits=(tp+fp)*(tp+fn)/n if n else 0; denom=tp+fp+fn-random_hits
    return (tp-random_hits)/denom if denom else 0.

def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true); p=np.asarray(y_pred); corr=np.corrcoef(y,p)[0,1] if len(y)>1 and np.std(y)>0 and np.std(p)>0 else float("nan")
    return {"mae":mean_absolute_error(y,p),"rmse":float(np.sqrt(mean_squared_error(y,p))),"r2":r2_score(y,p) if len(y)>1 else float("nan"),"bias":float(np.mean(p-y)),"correlation":float(corr)}
