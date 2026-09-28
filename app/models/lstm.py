"""Configurable PyTorch sequence model with per-grid sequence boundaries."""
from pathlib import Path
import numpy as np

class LSTMModel:
    def __init__(self,task="classification",sequence_length=24,hidden_size=32,epochs=15,patience=3,batch_size=64,learning_rate=1e-3,seed=42):
        try: import torch
        except ImportError as e: raise ImportError("Install the PyTorch extra: pip install -e '.[torch]'") from e
        if task not in {"classification","regression"}: raise ValueError("task must be classification or regression")
        if sequence_length < 1: raise ValueError("sequence_length must be positive")
        self.torch=torch; self.task=task; self.sequence_length=sequence_length; self.hidden_size=hidden_size; self.epochs=epochs; self.patience=patience; self.batch_size=batch_size; self.learning_rate=learning_rate; self.seed=seed; self.model=None

    @staticmethod
    def sequences(X,y,length,groups=None):
        """Create one left-padded history per row; sequences never cross group boundaries."""
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32)
        if len(X)!=len(y): raise ValueError("X and y lengths differ")
        group_values=np.zeros(len(X),dtype=int) if groups is None else np.asarray(groups)
        if len(group_values)!=len(X): raise ValueError("groups length differs from X")
        bounds=np.r_[0,np.flatnonzero(group_values[1:]!=group_values[:-1])+1,len(X)] if len(X) else [0]
        xs=[]; ys=[]
        for start,end in zip(bounds[:-1],bounds[1:]):
            for i in range(start,end):
                seq=X[max(start,i-length+1):i+1]
                if len(seq)<length: seq=np.vstack([np.repeat(seq[:1],length-len(seq),axis=0),seq])
                xs.append(seq); ys.append(y[i])
        return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32)

    def _build(self,input_size):
        torch=self.torch; nn=torch.nn; hidden=self.hidden_size
        class Net(nn.Module):
            def __init__(self):
                super().__init__(); self.lstm=nn.LSTM(input_size,hidden,batch_first=True); self.head=nn.Linear(hidden,1)
            def forward(self,x): return self.head(self.lstm(x)[0][:,-1]).squeeze(-1)
        self.model=Net()

    def fit(self,X,y,X_val=None,y_val=None,groups=None,val_groups=None):
        torch=self.torch; torch.set_num_threads(1); torch.manual_seed(self.seed)
        Xs,ys=self.sequences(X,y,self.sequence_length,groups); self._build(Xs.shape[-1])
        opt=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate); loss_fn=torch.nn.BCEWithLogitsLoss() if self.task=="classification" else torch.nn.MSELoss()
        tx=torch.tensor(Xs); ty=torch.tensor(ys); best=np.inf; wait=0; best_state=None
        if X_val is not None:
            vx,vy=self.sequences(X_val,y_val,self.sequence_length,val_groups); vxt=torch.tensor(vx); vyt=torch.tensor(vy)
        for _ in range(self.epochs):
            self.model.train()
            for ids in torch.randperm(len(tx)).split(self.batch_size):
                opt.zero_grad(); loss=loss_fn(self.model(tx[ids]),ty[ids]); loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad(): score=float(loss_fn(self.model(vxt),vyt)) if X_val is not None and len(vxt) else float(loss_fn(self.model(tx),ty))
            if score < best-1e-6: best=score; wait=0; best_state={k:v.clone() for k,v in self.model.state_dict().items()}
            else:
                wait+=1
                if wait>=self.patience: break
        if best_state: self.model.load_state_dict(best_state)
        return self

    def _outputs(self,X,groups=None):
        X=np.asarray(X,dtype=np.float32)
        seq,_=self.sequences(X,np.zeros(len(X),dtype=np.float32),self.sequence_length,groups)
        self.model.eval()
        with self.torch.no_grad(): return self.model(self.torch.tensor(seq)).numpy()

    def predict(self,X,groups=None):
        out=self._outputs(X,groups)
        return (1/(1+np.exp(-out))>=.5).astype(int) if self.task=="classification" else np.maximum(0,out)

    def predict_proba(self,X,groups=None):
        if self.task!="classification": raise ValueError("probability prediction only applies to classification")
        return 1/(1+np.exp(-self._outputs(X,groups)))

    def save(self,path,metadata=None):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        cfg={k:getattr(self,k) for k in ("task","sequence_length","hidden_size","epochs","patience","batch_size","learning_rate","seed")}
        self.torch.save({"state_dict":self.model.state_dict(),"model_config":cfg,"input_size":self.model.lstm.input_size,"metadata":metadata or {}},path)

    @classmethod
    def load(cls,path,map_location="cpu"):
        torch=__import__("torch")
        payload=torch.load(path,map_location=map_location,weights_only=False)
        obj=cls(**payload["model_config"]); obj._build(payload["input_size"])
        obj.model.load_state_dict(payload["state_dict"]); obj.model.eval()
        return obj
