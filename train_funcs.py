import torch
import numpy as np
from tqdm import tqdm
import ast

"""
TRAIN FUNCTION DEFINITION:
    train(model: StableDiffusionPipeline,
          projection_matrices: list[size=L](nn.Module),
          og_matrices: list[size=L](nn.Module),
          contexts: list[size=N](torch.tensor[size=MAX_LEN,...]),
          valuess: list[size=N](list[size=L](torch.tensor[size=MAX_LEN,...])),
          old_texts: list[size=N](str),
          new_texts: list[size=N](str),
          **kwargs)
    where L is the number of matrices to edit, and N is the number of sentences to train on (batch size).

PARAMS:
    model: the model to use.
    projection_matrices: list of projection matrices to edit from the model.
    og_matrices: list of original values for the projection matrices. detached from the model.
    contexts: list of context vectors (inputs to the matrices) to edit.
    valuess: list of results from all matrices for each context vector.
    old_texts: list of sentences to be edited.
    new_texts: list of target sentences to be aimed at.
    **kwargs: additional command line arguments.

TRAIN_FUNC_DICT defined at the bottom of the file.
"""

def baseline_train(model, projection_matrices, og_matrices, contexts, valuess, old_texts, new_texts):
    return None

def train_closed_form(ldm_stable, projection_matrices, og_matrices, contexts, valuess, old_texts,
          new_texts, layers_to_edit=None, lamb=0.1):
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    for layer_num in tqdm(range(len(projection_matrices))):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            #aggregate sums for mat1, mat2
            for context, values in zip(contexts, valuess):
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += for_mat1
                mat2 += for_mat2

            #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

TRAIN_FUNC_DICT = {
 "baseline": baseline_train,
 "train_closed_form": train_closed_form,
}