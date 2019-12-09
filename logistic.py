import warnings
from abc import ABCMeta,abstractmethod

import numpy as np
from scipy.sparse import issparse

from .base import BaseEstimator,ClassifierMixin

from .preprocessing import binarize
from sklearn.preprocessing import LabelBinarizer
from .preprocessing import label_binarize

from .utils import check_X_y,check_array,check_consistent_length
from .utils.extmath import safe_sparse_dot
from .utils.fixes import logsumexp
from .tuils.multiclass import _check_partitial_fit_first_call
from .tuils.validation import check_is_fitted
from .externals import six

__all__ = ['BernoulliNB','GaussianNB','MultinomialNB']
# 1 朴素贝叶斯基类
class BaseNB(six.with_metaclass(ABCMeta,BaseEstimator,ClassifierMixin)):
    @abstractmethod
    def _joint_log_likelihood(self,X):              # (1) 计算样本矩阵X的联合对数似然概率
                                                        # shape X = [N,C]

    def predict(self,X):                            # (2) 根据联合对数似然概率得出预测
        jll  = self._joint_log_likelihood(X)            # shape X = [N,C]
        return self.class_[np.argmax(jll,axis = 1)]     # shape jll = [N]
    def predict_log_proba(self,X):                  # (3) 根据联合对数似然概率计算样本的预测对数概率
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll,axis = 1 )           # shape log_prob_x = [N]
        return jll - np.atleast_2d(log_prob_x).T        # shape = [N,C]
    def predict_proba(self,X):                      # (4) 根据联合对数似然概率计算预测概率
        return np.exp(self.predict_log_proba(X))        # shape = [N,C]
# 2 高斯朴素贝叶斯类
class GaussianNB(BaseNB):
    def __init__(self,priors = None):               # (1) 属性：先验分布 priors
        self.priors = priors
    def fit(self,X,y,sample_weight = None):         # (2) 方法：拟合数据集，调用方法self._partial_fit
        X,y = check_X_y(X,y)
        return self._partial_fit(X,y,np.unique(y),
                                 _refit = True,
                                 sample_weight = sample_weight)
    @staticmethod
    def _update_mean_variance(n_past,mu,var,X,sample_weight = None):
                                                    # (3) 方法：动态更新均值和方差，以降低一次性内存消耗
        if X.shape[0] == 0:
            return mu,var
        if sample_weight is not None:
            n_new = float(sample_weight.sum())
            new_mu = np.average(X,axis = 0,weights = sample_weight / n_new) # 新均值 shape [C]
            new_var = np.average((X - new_mu) ** 2,                         # 新方差 shape [C]
                                 axis = 0,
                                 weights = sample_weight / n_new)
        else:
            n_new = X.shape[0]
            new_mu = np.mean(X,axis = 0)    # shape [C]
            new_var = np.var(X,axis = 0)    # shape [C]
        if n_past == 0:
            return new_mu,new_var

        n_total = float(n_past + n_new)
        total_mu = (n_new * new_mu + n_past * mu) / n_total
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd +
                     new_ssd +
                     (n_past / float(n_new * n_total)) * (n_new * mu - n_new * new_mu) ** 2
                     )
        total_var = total_ssd / n_total
        return total_mu,total_var
    def partial_fit(self,X,y,
                    classes = None,
                    sample_weight = None):
        return self._partial_fit(X,y,
                                 classes,
                                 _refit = False,
                                 sample_weight = sample_weight)
    def _partial_fit(self,X,y,
                     classes = None,
                     _refit = False,
                     sample_weight = None):
        X,y = check_X_y(X,y)                            # 输入检查
        if sample_weight is not None:                   # 输入检查
            sample_weight = check_array(sample_weight,
                                        ensure_2d = False)
            check_consistent_length(y,sample_weight)

        epsilon = le-9 * np.var(X,axis = 0).max()       # 防止数值病态

        if _refit:
            self.classes_ = None

        if _check_partitial_fit_first_call(self,classes):   # 首次调用
            n_features = X.shape[1]
            n_classes = len(self.classes_)                  # 类别个数
            self.theta_ = np.zeros((n_classes,n_features))  # 均值 shape = [C,D]
            self.sigma_ = np.zeros((n_classes,n_features))  # 均方差 shape = [C,D]
            self.class_count_ = np.zeros(n_classes,dtype = np.float64)

            n_classes = len(self.classes_)
            if self.priors is not None:                     # (1) 指定先验概率
                priors = np.asarray(self.priors)
                if len(priors) != n_classes:
                    raise ValueError('Number of priors must match number of classes.')
                if priors.sum() != 1.0:
                    raise ValueError('The sum of the priors should be 1.')
                if (priors < 0).any():
                    raise ValueError('Priors must be non-negative.')
                self.class_priors_ = priors                 # 类先验概率 shape = [C]
            else:                                           # (2) 未指定先验概率
                if X.shape[1] != self.theta_.shape[1]:
                    msg = 'Number of features %d does not match previous data %d.'
                    raise ValueError(msg % (X.shape[1],self.theta_.shape[1]))
                self.sigma_[:,:] -= epsilon
            
            classes = self.classes_

            unique_y = np.unique(y)
            unique_y_in_classes = np.inld(unique_y,classes)

            if not np.all(unique_y_in_classes):
                raise ValueError('The target label(s) %s in y do not exist in the initial classes % s'
                                % (unique_y[~unique_y_in_classes],classes))
            
            for y_i in unique_y:
                i = classes.searchsorted(y_i)
                X_i = X[y == y_i,:]

                if sample_weight is not None:
                    sw_i = sample_weight[y == y_i]
                    N_i = sw_i.sum()
                else:
                    sw_i = None
                    N_i = X_i.shape[0]
                
                new_theta,new_sigma = self._update_mean_variance(
                    self.class_count_[i],
                    self.theta_[i,:],
                    self.sigma_[i,:],
                    X_i,sw_i
                )
            
            self.sigma_[:,:] += epsilon
            # 如果未提供类先验，则更新
            if self.priors is None:
                self.class_prior_ = self.class_count_ / self.class_count_.sum()
            return self
    
    def _joint_log_likelihood(self,X):
        check_is_fitted(self,'classes_')

        X = check_array(X)
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])   # 类别 i 的对数先验概率
            n_ij = -0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i,:]))  #
            n_ij -= np.sum(
                ((X - self.theta_[i,:]) ** 2) /
                (self.sigma_[i,:]),
                axis = 1
            )                                       # 类别 i 的对数概率，shape = [N]
            joint_log_likelihood.append(jointi + n_ij)
                                                    # 类别 i 的联合对数似然概率 shape = [c = c + 1,N]
            joint_log_likelihood = np.array(joint_log_likelihood).T
                                                    # 转置，形状从 [C,N] 转置为 [N,C]
            return joint_log_likelihood

_ALPHA_MIN = 1e-10

class BaseDiscreteNB(BaseNB):

    # 1 更新类对数先验概率
    def _update_class_log_prior(self,class_prior = None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError('Number of priors must match number of classes.')
            slef.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            self.class_log_prior_ = (np.log(self.class_count_) -
                                    np.log(self.class_count_.sum()))
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)
    
    def _check_alpha(self):
        if self.alpha < 0:
            raise ValueError('Smoothing parameter alpha = %1.e '
                             'alpha should be > 0.' % self.alpha)
        if self.alpha < _ALPHA_MIN:
            warnings.warn('alpha too small will result in numeric errors,'
                          'alpha should be > 0.' %self.alpha)
            return _ALPHA_MIN
        return self.alpha
    
    def partial_fit(self,X,y,classes = None,sample_weight = None):
        X = check_array(X,accept_sparse = 'csr',dtype = np.float64)
        _,n_features = X.shape
        if _check_partitial_fit_first_call(self,classes):
            n_effective_classes = len(classes) if len(classes) > 1 else 2
            self.class_count_ = np.zeros(n_effective_classes,   # 类样本数 shape = [C]
                                        dtype = np.float64)
            self.feature_count_ = np.zeros((n_effecitve_classes,n_features),dtype = np.float64)
                                                                # 每个类的特征数 shape = [C,D]
        elif n_features != self.coef_.shape[1]:
            msg = 'Number of features %d does not match previous data %d.'
            raise ValueError(msg % (n_features,self.coef_.shape[-1]))
        
        Y = label_binarize(y,classes = self.classes_)
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y,Y),axis = 1)
        
        n_samples,n_classes = Y.shape

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0] %d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0],y.shape[0]))

        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T
        
        class_prior = self.class_prior

        self._count(X,Y)

        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior = class_prior)
        return self
    
    def fit(self,X,y,sample_weight = None):
        X,y = check_X_y(X,y,'csr')
        _,n_features = X.shape

        labelbin =LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y,Y),axis = 1)
        Y = Y.astype(np.float64)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y = check_array(sample_weight).T
        
        class_prior = self.class_prior

        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_features_classes,dtype = np.float64)
        self.feature_count_ = np.zeros((n_effective_classes,n_features),dtype = np.float64)
        self._count(X,Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior = class_prior)
        return self
    # 应急措施：正确设定类对数先验概率的维度，特征对数概率。
    def _get_coef(self):
        return (self.feature_log_prob_[1:]
                if len(self.classes_) == 2 else self.feature_log_prob_)
    def _get_intercept(self):
        return (self.class_log_prior_[1:]
                if len(self.classes_) == 2 else self.class_log_prior_)
    
    coef_ = property(_get_coef)
    intercept_ property(_get_intercept)

class MultinomialNB(BaseDiscreteNB):
    def __init__(self,alpha = 1.0,fit_prior = True,class_prior = None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
    def _count(self,X,Y):
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError('Input X must be non-negative')
        self.feature_count_ += safe_sparse_dot(Y.T,X)
        self.class_count_ += Y.sum(axis = 0)
    def _update_feature_log_prob(self,alpha):
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis = 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - 
                                 np.log(smoothed_cc.reshape(-1,1))
    def _joint_log_likelihood(self,X):
        check_is_fitted(self,"classes_")
        X = check_array(X,accept_sparse = 'csr')
        if self.binarize is not None:
            X = binarize(X,threshold = self.binarize)
        
        n_classes,n_features = self.feature_log_prob_.shape
        n_samples,n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError('Expected input with %d features,got %d instead'
                             % (n_features_X,n_features_X))
        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # 计算  neg_prob · (1 - X).T  为  ∑neg_prob - X · neg_prob
        jll = safe_sparse_dot(X,(self.feature_log_prob_ - neg_prob).T)
        jll += self.class_log_prior_ + neg_prob.sum(axis = 1)
        return jll
        

