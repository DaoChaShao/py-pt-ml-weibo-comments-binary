#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/29 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Data Processing Module - Deep Learning Workflow
----------------------------------------------------------------
This package provides utility modules for processing and preparing
datasets for machine learning, NLP, CV, and general data tasks.

Main Categories:
+ process_data : functions and classes for preprocessing raw data
+ prepare_data : functions and classes for preparing datasets for training and validation dataloaders

Usage:
+ Direct import from this package:
    - from src.configs import process_data, prepare_data
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.2.0"

from .preprocessor import process_data
from .setter import prepare_data

__all__ = [
    "process_data",
    "prepare_data",
]
