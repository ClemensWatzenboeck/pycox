

def my_test_print():
    print("Hello world!!!!")



def my_test_print_():
    print("-------------- --Hello world!!!!  ----------------")



# As remarked in this github issue:
#  https://github.com/havakv/pycox/issues/26
# We simply need to implement the corresponding API to utilize hyperparamter search. 
#  https://scikit-learn.org/stable/developers/develop.html
# Thé API we need to implement is:
#   estimator -> fit(X,y)
#   model     -> score(X,y)
# The basic version of the code is taken from:
#  https://gist.github.com/GCBallesteros/9d593b371dab1861c5e527617be40c0f


from sklearn.base import BaseEstimator
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models.cox import CoxPH
from pycox.models import LogisticHazard
import matplotlib.pyplot as plt




class DeepSURVSklearnAdapter(BaseEstimator):
    def __init__(
        self,
        learning_rate=1e-4,
        batch_norm=True,
        dropout=0.0,
        num_nodes=[32, 32],
        batch_size=128,
        epochs=10,
    ):
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        self.net_ = tt.practical.MLPVanilla(
            X.shape[1],
            self.num_nodes,
            1,
            self.batch_norm,
            self.dropout,
            output_bias=True,
        )
        self.deepsurv_ = CoxPH(self.net_, tt.optim.Adam)
        self.deepsurv_.optimizer.set_lr(self.learning_rate)

        # Sklearn needs the y inputs to be arranged as a matrix with each row
        # corresponding to an example but CoxPH needs a tuple with two arrays?
        y_ = (y[:, 0], y[:, 1])

        callbacks = [tt.callbacks.EarlyStopping()]
        log = self.deepsurv_.fit(
            X,
            y_,
            self.batch_size,
            self.epochs,
            verbose=False,
        )

        return self

    def score(self, X, y):
        _ = self.deepsurv_.compute_baseline_hazards()
        surv = self.deepsurv_.predict_surv_df(X)

        ev = EvalSurv(
            surv,
            y[:, 0],  # time to event
            y[:, 1],  # event
            censor_surv="km",
        )

        return ev.concordance_td()

#------------------------------------------------------

class LogisticHazardSklearnAdapter(BaseEstimator):
    def __init__(
        self,
        learning_rate=1e-4,
        batch_norm=True,
        dropout=0.0,
        num_nodes=[32, 32],
        batch_size=512,
        epochs=10,
        num_durations = 20,  # number of timegrid points in output
        label_transform_scheme = 'quantiles',
        interpolationsteps = 100
    ):
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.num_durations = num_durations
        self.label_transform_scheme = label_transform_scheme
        self.batch_size = batch_size
        self.epochs = epochs
        self.interpolationsteps = interpolationsteps

    def fit(self, X: pd.DataFrame, y: np.recarray):
        """
        y needs to have the fields 'time' and 'status'
        """

        X_ = X.astype(np.float32)#.to_numpy()

        self.labtrans = LogisticHazard.label_transform(self.num_durations, self.label_transform_scheme)
        y_ = self.labtrans.fit_transform(y.time, y.status)
        self.out_features = self.labtrans.out_features

        self.net_ = tt.practical.MLPVanilla(
            X_.shape[1],
            self.num_nodes,
            self.out_features,
            self.batch_norm,
            self.dropout,
            # output_bias=True,
        )

        self.optimizer = tt.optim.AdamWR(decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8,
                            cycle_multiplier=2)
        self.model_ = LogisticHazard(self.net_, self.optimizer, duration_index=self.labtrans.cuts)

        # instead of learning rate finder: 
         # lrfind = self.model_.lr_finder(x_train, y_train, batch_size, tolerance=50)

        self.model_.optimizer.set_lr(self.learning_rate)

        callbacks = [tt.callbacks.EarlyStopping()] # only makes sense with val. set I guess
        log = self.model_.fit(
            X_,
            y_,
            self.batch_size,
            self.epochs,
            #callbacks,
            verbose=False,
        )

        #_ = log.to_pandas().iloc[1:].plot()
        #plt.ylim(0,10)
        #plt.grid()
        #plt.show()

        return self

    def score(self, X, y):
        X_ = X.astype(np.float32)#.to_numpy()
        #_ = self.model_.compute_baseline_hazards()
        surv = self.model_.interpolate(self.interpolationsteps).predict_surv_df(X_)

        ev = EvalSurv(
            surv,
            y.time,  # time to event
            y.status,  # event
            censor_surv="km",
        )

        return ev.concordance_td()
#-------------------

