from models.model import Model
from data.cai import CAI
import json


def process_datapoint(data_point):

    harmful_prompt = data_point['init_prompt']
    undesirable_response = data_point['init_response']
    better_response = data_point['revision_response']
    return harmful_prompt, undesirable_response, better_response


def generate_principle(model, user_prompt):
    return model.query_gpt(system_prompt="", user_prompt=user_prompt)


def generate_principles(data_points, model, num_clusters):
    principles = []
    for data_point in data_points:
        harmful_prompt, undesirable_response, better_response  = process_datapoint(data_point)
        user_prompt = f"{harmful_prompt} Undesirable Response: {undesirable_response} Better Response: {better_response}"
        principle = generate_principle(model, user_prompt)
        principles.append(principle)  # change this line to add_principle_to_principles(principle), which also calls entailment
    return cluster_principles(principles, num_clusters)


def cluster_principles(principles, num_clusters):
    return principles  # lol not sure what to do here, for now just returning all the principles but ideally we form num_cluster clusters, and then create a summary principle for each
