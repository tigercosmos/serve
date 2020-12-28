"""
Util files for TorchServe
"""
import inspect
import os
import json
import itertools
import logging


from collections import OrderedDict
import types
import ast
import copy
import textwrap
import astunparse

import torch
from torch.nn.modules import Module
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)

logger = logging.getLogger(__name__)

def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.

    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                    member.__module__ == module.__name__)]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes

def load_label_mapping(mapping_file_path):
    """
    Load a JSON mapping { class ID -> friendly class name }.
    Used in BaseHandler.
    """
    if not os.path.isfile(mapping_file_path):
        logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')
        return None

    with open(mapping_file_path) as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise Exception('index_to_name mapping should be in "class":"label" json format')

    # Older examples had a different syntax than others. This code accommodates those.
    if 'object_type_names' in mapping and isinstance(mapping['object_type_names'], list):
        mapping = {str(k): v for k, v in enumerate(mapping['object_type_names'])}
        return mapping

    for key, value in mapping.items():
        new_value = value
        if isinstance(new_value, list):
            new_value = value[-1]
        if not isinstance(new_value, str):
            raise Exception('labels in index_to_name must be either str or [str]')
        mapping[key] = new_value
    return mapping

def map_class_to_label(probs, mapping=None, lbl_classes=None):
    """
    Given a list of classes & probabilities, return a dictionary of
    { friendly class name -> probability }
    """
    if not (isinstance(probs, list) and isinstance(probs, list)):
        raise Exception('Convert classes to list before doing mapping')
    if mapping is not None and not isinstance(mapping, dict):
        raise Exception('Mapping must be a dict')

    if lbl_classes is None:
        lbl_classes = itertools.repeat(range(len(probs[0])), len(probs))

    results = [
        {
            (mapping[str(lbl_class)] if mapping is not None else str(lbl_class)): prob
            for lbl_class, prob in zip(*row)
        }
        for row in zip(lbl_classes, probs)
    ]

    return results

# https://github.com/rniczh/torch-model-split
class _CudaMappingVisitor(ast.NodeVisitor):
    def __init__(self, output_device=None, layer_gpus=OrderedDict()):
        super(_CudaMappingVisitor)
        self.layer_gpus = layer_gpus
        self.output_device = output_device

    def visit_Return(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        # processing return val => return val.cuda(output_device)
        # in AST
        #   Return(value=val)
        # =>
        #   Return(value=Call(func=Attribute(value=arg,
        #                                  attr='cuda',
        #                                  ctx=Load()),
        #                   args=[Num(n=output_device)]))
        value = ast.Call(func=ast.Attribute(value=node.value,
                                            attr='cuda',
                                            ctx=ast.Load()),
                         args=[ast.Num(n=self.output_device)],
                         keywords=[], starargs=None, kwargs=None)
        node.value = value


    #! currently, it only can deal with the layer call with only one argument
    def visit_Call(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        # processing self.layer(arg) => self.layer(arg.cuda(device_id))
        # In AST
        #   Call(func=Attribute(value=Name(id='self', ctx=Load()),
        #                       attr ='layer',
        #                       ctx  =Load()),
        #        args=[arg])
        # =>
        #   Call(func=Attribute(value=Name(id='self', ctx=Load()),
        #                       attr ='layer',
        #                       ctx  =Load()),
        #        args=[Call(func=Attribute(value=arg,
        #                                  attr='cuda',
        #                                  ctx=Load()),
        #                   args=[Num(n=device_id)]
        #             ])
        func = node.func
        if (len(node.args) == 1 and  # TODO, release the restrict of one argument
            isinstance(func, ast.Attribute) and
            isinstance(func.ctx, ast.Load)  and
            isinstance(func.value, ast.Name)):
            value = func.value
            attr  = func.attr
            arg   = node.args[0]

            # check weather it is belong to model
            if value.id == 'self' and isinstance(value.ctx, ast.Load):
                # get the layer device id
                device_id = self.layer_gpus[attr]
                new_arg=ast.Call(func=ast.Attribute(value=arg,
                                                    attr='cuda',
                                                    ctx=ast.Load()),
                                 args=[ast.Num(n=device_id)],
                                 keywords=[], starargs=None, kwargs=None)
                # udpate args
                node.args = [new_arg]

# https://github.com/rniczh/torch-model-split
class DataFlow(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, inference_only=False, clear_cache=True):
        super(DataFlow, self).__init__()

        device_type = _get_available_device_type()

        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])
        self.clear_cache = clear_cache

        # because inference only, so disable the gradient in model
        if inference_only:
            for param in self.module.parameters():
                param.requires_grad=False

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

        self.layer_gpus = OrderedDict()
        for name, module in self.module.named_children():
            self.layer_gpus[name] = self.output_device

        self.old_forward = copy.deepcopy(self.module.forward)

    def update_flow(self):
        self.module.forward = self.old_forward

        # update the submodule gpus
        for name, module in self.module.named_children():
            module.cuda(self.layer_gpus[name])

        if self.clear_cache:
            torch.cuda.empty_cache()

        # get the forward source code and convert it into AST
        source = textwrap.dedent(inspect.getsource(self.module.forward))
        tree = ast.parse(source)

        # udpate the AST
        v = _CudaMappingVisitor(layer_gpus=self.layer_gpus,
                                output_device=self.output_device)
        v.visit(tree)
        ast.fix_missing_locations(tree)

        # recompile
        code = compile(tree, filename="<ast>", mode="exec")
        namespace = self.module.forward.__globals__
        exec(code, namespace)
        self.module.forward = types.MethodType(namespace['forward'], self.module)

        # print(astunparse.unparse(tree))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
