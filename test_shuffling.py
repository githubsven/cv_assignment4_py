#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:50:42 2018

@author: breixo
"""

import imgUtils
import random

def count_items(a_list, size):
    count = [0]*size
    for item in a_list:
        count[item] += 1
    return count

def print_percent_items(count, n_items):
    count_copy = count.copy()
    for i in range(len(count_copy)):
        count_copy[i] = count_copy[i]/n_items
    print(count_copy)

def print_results(a_list, size):
    n_images = len(a_list)
    count = count_items(a_list, size)
    print_percent_items(count, n_images)

def main():
    size = 5
    
    images_sorted = list(range(size))
    labels_sorted = list(range(size -1, -1, -1))
    
    results_images = dict()
    results_labels = dict()
    
    for i in range(size):
        results_images[i] = []
        results_labels[i] = []
    
    for i in  range(10000):
        images = images_sorted.copy()
        labels = labels_sorted.copy()
        imgUtils.shuffle(images, labels)
        for i in range(size):
            results_images[i].append(images[i])
            results_labels[i].append(labels[i])
    
    for i in range(size):
        list_images = results_images[i]
        print_results(list_images, size)
    for i in range(size):
        list_labels = results_labels[i]
        print_results(list_labels, size)

if __name__ == "__main__":
    main()