# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:31:20 2023

@author: Nguyen Minh Thuan
"""

from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
import pandas as pd

class FeatureSelector(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # initialize sel_col
        self.sel_col = None
    
    def GA(self):
        """
        feature selection following Genetic Algorithms
        """
        # import GA library
        from feature_selection.FS.ga import jfs
        
        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 15    # number of chromosomes
        T    = 30   # maximum number of generations
        CR   = 0.8
        MR   = 0.01
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

        # perform genetic algorithm for feature selection
        print("Start preforming genetic algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        print("Curve is:", fmdl['c'])
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def PSO(self):
        """
        feature selection following the PSO algorithm
        """
        from feature_selection.FS.pso import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameter
        k    = 5     # k-value in KNN
        N    = 30    # number of particles
        T    = 30   # maximum number of iterations
        w    = 0.9
        c1   = 2
        c2   = 2
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}

        # perform the Particle Swarm Optimization for feature selection
        print("Start preforming the Particle Swarm Optimization for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def WOA(self):
        """
        feature selection following the WOA algorithm
        """
        from feature_selection.FS.woa import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     
        N    = 30    
        T    = 30   
        b  = 1    
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b}

        # perform the Whale Optimization Algorithm for feature selection
        print("Start preforming the Whale Optimization Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        print("Curve is:", fmdl['c'])
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def WOA_GA(self):
        """
        feature selection following the WOA_GA algorithm
        """
        from feature_selection.FS.woa_ga import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Whale Optimization Algorithm
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     
        N    = 30    
        T    = 30  
        b    = 1
        CR   = 0.8
        MR   = 0.1    
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'b':b, 'CR':CR, 'MR':MR}

        # perform the Genetic Whale Optimization Algorithm for feature selection
        print("Start preforming the Genetic Whale Optimization Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        print(f"feature collumns: {self.sel_col}")
        print("Curve is:", fmdl['c'])
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def BA(self):
        """
        feature selection following the BAT algorithm
        """
        from feature_selection.FS.ba import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 30    # number of particles
        T    = 30   # maximum number of iterations
        fmax   = 2      # maximum frequency
        fmin   = 0      # minimum frequency
        alpha  = 0.9    # constant
        gamma  = 0.9    # constant
        A      = 2      # maximum loudness
        r      = 1      # maximum pulse rate
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'fmax':fmax, 'fmin':fmin, 'alpha':alpha, 'gamma':gamma, 'A':A, 'r':r}

        # perform the Bat Algorithm for feature selection
        print("Start preforming the Bat Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        print("Curve is:", fmdl['c'])
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def CS(self):
        """
        feature selection following the Cuckoo Search
        """
        from feature_selection.FS.cs import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 10    # number of particles
        T    = 100   # maximum number of iterations
        Pa  = 0.25   # discovery rate
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'Pa':Pa}

        # perform the Cuckoo Search for feature selection
        print("Start preforming the Cuckoo Search for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def DE(self):
        """
        Feature selection following Differential Evolution
        """
        from feature_selection.FS.de import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 10    # number of particles
        T    = 100   # maximum number of iterations
        CR = 0.9    # crossover rate
        F  = 0.5    # constant factor
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'F':F}

        # perform the Differential Evolution for feature selection
        print("Start preforming the Differential Evolution for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test

    def FA(self):
        """
        feature selection following the Firefly Algorithm
        """
        from feature_selection.FS.fa import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 10    # number of particles
        T    = 100   # maximum number of iterations
        alpha  = 1       # constant
        beta0  = 1       # light amplitude
        gamma  = 1       # absorbtion coefficient
        theta  = 0.97    # control alpha
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'alpha':alpha, 'beta0':beta0, 'gamma':gamma, 'theta':theta}

        # perform the Firefly Algorithm for feature selection
        print("Start preforming the Firefly Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def FPA(self):
        """
        feature selection following the Flower Pollination Algorithm
        """
        from feature_selection.FS.fpa import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Genetic Algorithms
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 10    # number of particles
        T    = 100   # maximum number of iterations
        P  = 0.8      # switch probability
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'P':P}

        # perform the Flower Pollination Algorithm for feature selection
        print("Start preforming the Flower Pollination Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test
    
    def SCA(self):
        """
        feature selection following the Sine Cosine Algorithm
        """
        from feature_selection.FS.sca import jfs

        # convert dataframe to numpy array
        feat = np.asarray(self.X_train)
        label = np.asarray(self.y_train)

        # prepare data feadfoward to Sine Cosine Algorithm
        fold = {'xt': self.X_train, 'yt':self.y_train, 'xv':self.X_test, 'yv':self.y_test}

        # parameters
        k    = 5     # k-value in KNN
        N    = 30    # number of particles
        T    = 30   # maximum number of iterations
        alpha  = 2    # constant
        opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'alpha':alpha}

        # perform the Sine Cosine Algorithm for feature selection
        print("Start preforming the Sine Cosine Algorithm for feature selection")
        fmdl = jfs(feat, label, opts)
        self.sel_col = fmdl['sf']
        print("Curve is:", fmdl['c'])
        return self.X_train.iloc[:, self.sel_col], self.y_train, self.X_test.iloc[:, self.sel_col], self.y_test







        





