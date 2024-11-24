#!/usr/bin/env python3

# Dataset root (absolute or relative)
dataset_root = "./data"

# YAML content
yaml_content = f"""
train: {dataset_root}/images/train
val: {dataset_root}/images/val

nc: 60  # Number of classes

names: [
    "Aluminium foil", 
    "Battery", 
    "Aluminium blister pack", 
    "Carded blister pack", 
    "Other plastic bottle", 
    "Clear plastic bottle", 
    "Glass bottle", 
    "Plastic bottle cap", 
    "Metal bottle cap", 
    "Broken glass", 
    "Food Can", 
    "Aerosol", 
    "Drink can", 
    "Toilet tube", 
    "Other carton", 
    "Egg carton", 
    "Drink carton", 
    "Corrugated carton", 
    "Meal carton", 
    "Pizza box", 
    "Paper cup", 
    "Disposable plastic cup", 
    "Foam cup", 
    "Glass cup", 
    "Other plastic cup", 
    "Food waste", 
    "Glass jar", 
    "Plastic lid", 
    "Metal lid", 
    "Other plastic", 
    "Magazine paper", 
    "Tissues", 
    "Wrapping paper", 
    "Normal paper", 
    "Paper bag", 
    "Plastified paper bag", 
    "Plastic film", 
    "Six pack rings", 
    "Garbage bag", 
    "Other plastic wrapper", 
    "Single-use carrier bag", 
    "Polypropylene bag", 
    "Crisp packet", 
    "Spread tub", 
    "Tupperware", 
    "Disposable food container", 
    "Foam food container", 
    "Other plastic container", 
    "Plastic glooves", 
    "Plastic utensils", 
    "Pop tab", 
    "Rope & strings", 
    "Scrap metal", 
    "Shoe", 
    "Squeezable tube", 
    "Plastic straw", 
    "Paper straw", 
    "Styrofoam piece", 
    "Unlabeled litter", 
    "Cigarette"
]
"""

# Write to a YAML file
with open("data.yaml", "w") as yaml_file:
    yaml_file.write(yaml_content)

print("Generated data.yaml")
