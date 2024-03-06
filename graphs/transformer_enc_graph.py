from graphs.base_graph import BIGGraph, NodeType
import torch

class TransformerEncoderGraph(BIGGraph):
    
    def __init__(self, model,
                 modules,
                 layer_name='', # for transformer
                 enc_prefix='encoder',
                 merge_type='ff_only',
                 num_layers=12,
                 num_heads=8,
                 qk=False,
                 name='bert',
                 classifier=False):
        super().__init__(model)
        
        self.layer_name = layer_name
        self.enc_prefix = enc_prefix
        self.merge_type = merge_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.modules = modules
        self.qk = qk
        self.name = name
        self.classifier = classifier 


    def add_layerblock_nodes(self, name_prefix, input_node, merge_type):
        # first half
        modules = self.modules
        # do attention block here
        residual = input_node
        value_node = self.add_nodes_from_sequence(name_prefix, [modules['v']], residual)
        if self.qk:
            key_node = self.add_nodes_from_sequence(name_prefix, [modules['k'], NodeType.POSTFIX], residual)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.POSTFIX, NodeType.SUM], residual)
        else:
            key_node = self.add_nodes_from_sequence(name_prefix, [modules['k']], residual)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['q'], NodeType.SUM], residual)
        self.add_directed_edge(key_node, input_node) # add key to "SUM" - it is really just a product but same handler
        input_node = self.add_nodes_from_sequence(name_prefix, [NodeType.SUM], input_node) #sum (mult)node to outproj
        self.add_directed_edge(value_node, input_node) #value node to sum (mult)
        
        if merge_type == 'ff_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node)
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        if merge_type == 'res_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node)
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        elif merge_type == 'ff+res':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get first residual vector from after self attn layer norm
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        elif merge_type == 'ff+attn':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        elif merge_type == 'attn_only':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln']], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        elif merge_type == 'res+attn':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)


        elif merge_type == 'all':
            # add self attn out proj to dot prod, layer norm, sum residual
            # get intermeds between attn and self attn out proj
            # get first residual vector from after self attn layer norm
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                    [NodeType.PREFIX, modules['lin_attn'], NodeType.SUM], 
                                                    input_node) 
            # add & norm
            self.add_directed_edge(residual, input_node)
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['attn_ln'], NodeType.POSTFIX], input_node=input_node)

            # do second half with residual too
            residual = input_node
            input_node = self.add_nodes_from_sequence(name_prefix, 
                                                  [modules['fc1'], NodeType.PREFIX, modules['fc2'], NodeType.SUM], 
                                                  input_node=input_node)
            self.add_directed_edge(residual, input_node)

        if merge_type in ['all', 'ff+res', 'res_only', 'res+attn']:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln'], NodeType.POSTFIX], input_node=input_node)
        else:
            input_node = self.add_nodes_from_sequence(name_prefix, [modules['final_ln']], input_node=input_node)
        return input_node

    def add_layer_nodes(self, layer_prefix, input_node, merge_type):
        source_node = input_node
        
        for layer_index in range(self.num_layers): # for graph visualization
        #for layer_index, layerblock in enumerate(self.get_module(name_prefix)):
            source_node = self.add_layerblock_nodes(layer_prefix+f'.{layer_index}', source_node, merge_type)        
        return source_node

    def graphify(self):
        modules = self.modules
        # keep input node
        input_node = self.create_node(node_type=NodeType.INPUT)
        # input_node -> emb_tok 
        emb_name = modules['emb']
        emb_node = self.create_node(node_type=NodeType.EMBEDDING, 
                                    layer_name=f'{self.enc_prefix}.{emb_name}'.strip('.'),
                                    param_name=f'{self.enc_prefix}.{emb_name}.weight'.strip('.'))
        self.add_directed_edge(input_node, emb_node)

        # removing emb_pos node for now...
        input_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['emb_ln']], emb_node) 
     
        if self.merge_type in ['all', 'ff+res', 'res_only']:
            #adding postfix to emb_ln, before xformer layers
            input_node = self.add_nodes_from_sequence(self.enc_prefix, [NodeType.POSTFIX], input_node)

        # layernorm_embedding -> xformer layers
        input_node = self.add_layer_nodes(f'{self.layer_name}', input_node, self.merge_type)
                
        # xformer layers -> dense -> layernorm -> output
        if self.name == 'bert' and self.classifier == False:
            dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['transform.dense', 'transform.LayerNorm', NodeType.PREFIX, 'decoder'], input_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(dense_node, output_node)
        elif self.name == 'bert' and self.classifier == True:
            pool_node = self.add_nodes_from_sequence(self.enc_prefix, [modules['pooler']], input_node)
            class_node = self.add_nodes_from_sequence('', [NodeType.PREFIX, modules['classifier']], pool_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(class_node, output_node)
        elif self.name == 'roberta':
            #dense_node = self.add_nodes_from_sequence(modules['head_pref'], ['dense', NodeType.PREFIX, 'out_proj'], input_node)
            output_node = self.create_node(node_type=NodeType.OUTPUT)
            self.add_directed_edge(input_node, output_node)       
        
        return self

    
def bert(model, merge_type='ff_only', qk=False, classifier=False):
    modules = {'emb': 'embeddings.word_embeddings',
     'emb_pos': 'embeddings.position_embeddings',
     'emb_tok_type': 'embeddings.token_type_embeddings',
     'emb_ln': 'embeddings.LayerNorm',
     'q': 'attention.self.query',
     'k': 'attention.self.key',
     'v': 'attention.self.value',
     'lin_attn': 'attention.output.dense',
     'attn_ln': 'attention.output.LayerNorm',
     'fc1': 'intermediate.dense',
     'fc2': 'output.dense',
     'final_ln': 'output.LayerNorm',
     'head_pref': 'cls.predictions',
     'pooler': 'pooler.dense',
     'classifier': 'classifier'}
    return TransformerEncoderGraph(model, 
                                   modules,
                                   layer_name='bert.encoder.layer', 
                                   enc_prefix='bert',
                                   merge_type=merge_type,
                                   num_layers=12,
                                   num_heads=12,
                                   qk=qk,
                                   name='bert',
                                   classifier=classifier)



'''
checks if two state_dicts are the same. Used for debugging purposes.
reference: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212 
'''
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda":
            v_2 = v_2.to("cuda" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2, atol=1e-03):
            print(k_1)
            print(f"Tensor mismatch: {v_1} vs {v_2}")


