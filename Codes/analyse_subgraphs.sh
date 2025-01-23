#!/bin/bash

#analyse the subgraphs of the model to see where it runs CPU/DPU
 
xir    graph ./compiled/Compiled_model.xmodel 2>&1 | tee Compiled_model_graph_info.txt
xir subgraph ./compiled/Compiled_model.xmodel 2>&1 | tee Compiled_model_subgraph_tree.txt
xir dump_txt ./compiled/Compiled_model.xmodel            Compiled_model_dump_xmodel.txt
xir png      ./compiled/Compiled_model.xmodel           Compiled_model_xmodel.png
xir svg      ./compiled/Compiled_model.xmodel           Compiled_model_xmodel.svg