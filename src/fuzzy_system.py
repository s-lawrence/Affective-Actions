#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:24:03 2021

@author: s7lawren
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class FuzzySystem():
    def __init__(self):
        # Universe Vaiables
        self.x_cluttered = np.arange( 0, 11, 1   )
        self.x_uncertain = np.arange( 0,  1, 0.05)
        self.x_emotion   = np.arange(-1,  1, 0.05)
        
        # Generate fuzzy memberships functions
        self.cluttered_lo = fuzz.trimf(self.x_cluttered, [0 ,0  ,2 ])
        self.cluttered_md = fuzz.trimf(self.x_cluttered, [0 ,2  ,5])
        self.cluttered_hi = fuzz.trimf(self.x_cluttered, [2 ,10 ,10])
        
        self.uncertain_lo = fuzz.trimf(self.x_uncertain, [0   ,0   ,0.5])
        self.uncertain_md = fuzz.trimf(self.x_uncertain, [0   ,0.5 ,1  ])
        self.uncertain_hi = fuzz.trimf(self.x_uncertain, [0.5 ,1   ,1  ])
        
        self.emotion_lo   = fuzz.trimf( self.x_emotion, [-1,   -1,   0 ])
        self.emotion_md   = fuzz.trapmf(self.x_emotion, [-0.3, -0.08, 0.08, 0.3])
        self.emotion_hi   = fuzz.trimf( self.x_emotion, [ 0,    1,   1 ])

    def plot_membersip_functions(self):
        # Visualize these universes and membership functions
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
        
        ax0.plot(self.x_cluttered, self.cluttered_lo, 'b', linewidth=1.5, label='Low'     )
        ax0.plot(self.x_cluttered, self.cluttered_md, 'g', linewidth=1.5, label='Moderate')
        ax0.plot(self.x_cluttered, self.cluttered_hi, 'r', linewidth=1.5, label='High'    )
        ax0.set_title('Cluttered')
        ax0.legend()
        
        ax1.plot(self.x_uncertain, self.uncertain_lo, 'b', linewidth=1.5, label='Low'     )
        ax1.plot(self.x_uncertain, self.uncertain_md, 'g', linewidth=1.5, label='Moderate')
        ax1.plot(self.x_uncertain, self.uncertain_hi, 'r', linewidth=1.5, label='High'    )
        ax1.set_title('Uncertainty of Objects')
        ax1.legend()
        
        ax2.plot(self.x_emotion, self.emotion_lo, 'b', linewidth=1.5, label='Stressed')
        ax2.plot(self.x_emotion, self.emotion_md, 'g', linewidth=1.5, label='Neutral' )
        ax2.plot(self.x_emotion, self.emotion_hi, 'r', linewidth=1.5, label='Happy'   )
        ax2.set_title('Emotion')
        ax2.legend()
        
        # Turn off top/right axes
        for ax in (ax0, ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        
        plt.tight_layout()

    def interpret(self, clutter, uncertainty):
        clutter_level_lo = fuzz.interp_membership(self.x_cluttered, self.cluttered_lo, clutter)
        clutter_level_md = fuzz.interp_membership(self.x_cluttered, self.cluttered_md, clutter)
        clutter_level_hi = fuzz.interp_membership(self.x_cluttered, self.cluttered_hi, clutter)
        
        uncertain_level_lo = fuzz.interp_membership(self.x_uncertain, self.uncertain_lo, uncertainty)
        uncertain_level_md = fuzz.interp_membership(self.x_uncertain, self.uncertain_md, uncertainty)
        uncertain_level_hi = fuzz.interp_membership(self.x_uncertain, self.uncertain_hi, uncertainty)
        # Rule for high clutter or high uncertainty 
        active_rule1 = np.fmax(np.fmax(clutter_level_hi, uncertain_level_hi), 
                               np.fmin(clutter_level_md, uncertain_level_md))
        self.emotion_activation_lo = np.fmin(active_rule1, self.emotion_lo)
        
        active_rule2 = np.fmax(np.fmin(clutter_level_lo, uncertain_level_md),
                               np.fmin(clutter_level_md, uncertain_level_lo))
        self.emotion_activation_md = np.fmin(active_rule2, self.emotion_md)
        
        active_rule3 = np.fmin(clutter_level_lo, uncertain_level_lo)
        self.emotion_activation_hi = np.fmin(active_rule3, self.emotion_hi)
        
        # Aggregate all three output membership functions together
        aggregated = np.fmax(self.emotion_activation_lo,
                             np.fmax(self.emotion_activation_md, self.emotion_activation_hi))
        
        # Calculate defuzzified result
        self.emotion = fuzz.defuzz(self.x_emotion, aggregated, 'centroid')
        return self.emotion

    def plot_interpretation(self):
        # Visualize this
        emotion0 = np.zeros_like(self.x_emotion)
        fig, ax0 = plt.subplots(figsize=(8, 3))
        ax0.fill_between(self.x_emotion, emotion0, self.emotion_activation_lo, facecolor='b', alpha=0.7)
        ax0.plot(self.x_emotion, self.emotion_lo, 'b', linewidth=0.5, linestyle='--', label="Stressed")
        ax0.fill_between(self.x_emotion, emotion0, self.emotion_activation_md, facecolor='g', alpha=0.7)
        ax0.plot(self.x_emotion, self.emotion_md, 'g', linewidth=0.5, linestyle='--', label="Neutral")
        ax0.fill_between(self.x_emotion, emotion0, self.emotion_activation_hi, facecolor='r', alpha=0.7)
        ax0.plot(self.x_emotion, self.emotion_hi, 'r', linewidth=0.5, linestyle='--', label="Happy")
        ax0.set_title('Emotional Activity')
        ax0.legend()
        
        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
        
        plt.tight_layout()




